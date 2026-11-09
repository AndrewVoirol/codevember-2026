// Zero-Allocation WebGPU 3D Engine & Kinematic Solver for the Grand Coaxial, Da Vinci & Antikythera Orrery

export const GROUP_INFO = {
  1: {
    id: 1,
    code: "drive.*",
    name: "Handwheel Crank, Worm Barrel & Da Vinci Conical Fusee",
    summary: "4-spoke knurled handwheel drives a 4.5-start helical worm screw & 28T worm wheel, paired with a Da Vinci conical helical fusee drum equalizing mainspring barrel tension.",
  },
  2: {
    id: 2,
    code: "transfer.*",
    name: "Pillow-Block Line Shafts & Da Vinci Lantern-Crown Drives",
    summary: "Steel transmission shafts on cast-brass pillow blocks with 10-stave Da Vinci lantern pinions ('rocchetti a lanterna') meshing at 90° with contrate crown wheels & miter bevel gears.",
  },
  3: {
    id: 3,
    code: "gear.lay.*",
    name: "Countershaft & 6-Level Stepped Driving Pyramid",
    summary: "Front-window layshaft at [-13.2, -23.43] carrying six involute driving gears stepping from 62T Mercury at the bottom to 10T Saturn at the top with zero overhang occlusion.",
  },
  4: {
    id: 4,
    code: "turret.*",
    name: "6-Level Coaxial Driven Wheels & True Telescoping Tubes",
    summary: "Six central spur gears (16T, 30T, 40T, 52T, 58T, 64T) pinned to six concentric brass/steel tubes ordered bottom-to-top so no tube ever intersects another gear's web.",
  },
  5: {
    id: 5,
    code: "orrery.sun",
    name: "Central Rotating Sun & Heliocentric Lamp",
    summary: "Central brass-filigree solar sphere rotating with a 25.38-day sidereal period and casting real-time 3D day/night terminators and eclipse shadows across all planets.",
  },
  6: {
    id: 6,
    code: "arm.M",
    name: "Mercury Carrier Arm & 7.0° Inclined Orbit",
    summary: "Innermost tallest sleeve (z = 86.5, R = 17.5) driven at 62:16 (4.152× Earth rate, 87.97 d period) with counterweight bob and 7.00° orbital inclination.",
  },
  7: {
    id: 7,
    code: "arm.V",
    name: "Venus Carrier Arm & 3.39° Inclined Orbit",
    summary: "Second coaxial sleeve (z = 82.0, R = 29.5) driven at 48:30 (1.626× Earth rate, 224.70 d period) with counterweight bob and 3.39° orbital inclination.",
  },
  8: {
    id: 8,
    code: "arm.E + moon.*",
    name: "Earth Tellurian (5-Gear Parallel Train) & Antikythera Lunar Anomaly",
    summary: "Tooth-to-tooth 5-gear (24T×5) underside train keeps Earth's 23.44° axis pointed at Polaris; Antikythera pin-and-slot coupler & 5.14° nodal cam ring drive the Moon.",
  },
  9: {
    id: 9,
    code: "arm.Mars",
    name: "Mars Carrier Arm & 25.19° Obliquity Globe",
    summary: "Fourth sleeve (z = 73.0, R = 62.0) driven at 28:52 (0.531× Earth rate, 686.98 d period) with 25.19° axial obliquity and 1.85° orbital inclination.",
  },
  10: {
    id: 10,
    code: "drum.ecliptic",
    name: "6-Column Greek Zodiac Parapegma & Archimedes Armillary Arch",
    summary: "Six 16-flute Corinthian bronze columns supporting the 230° graduated Zodiac Parapegma ring and overarching Celestial Equinoctial & Solstitial Armillary arches.",
  },
  11: {
    id: 11,
    code: "ephem.decl",
    name: "Scotch-Yoke Declination Gauge & Equation-of-Time Kidney Cam",
    summary: "Slotted Scotch yoke converts circular motion λ_E into vertical travel δ = +23.44° sin(λ_E), beside a reniform Kidney Cam & jeweled lever computing Δt.",
  },
  12: {
    id: 12,
    code: "ephem.dial",
    name: "Antikythera Spiral Saros & 4-Hand Astronomical Dial",
    summary: "Engraved 4-turn Archimedes spiral Saros dial with coaxial Breguet Mean Sun, True Sun (Δt), Draconic Node (Ω/☋), and sliding spiral rider hands.",
  },
  13: {
    id: 13,
    code: "stand.*",
    name: "Machined Bedplate & Rack-and-Pinion Linear Ephemeris",
    summary: "Cast bedplate on leveled brass bun feet with a 54-tooth linear brass rack, 14T driving pinion, and jeweled Vernier cursor tracking day of the year.",
  },
  14: {
    id: 14,
    code: "davinci.escapement",
    name: "Da Vinci Crown-Wheel & Verge-Foliot Regulator (Codex Madrid I)",
    summary: "20-tooth contrate crown wheel engaging a vertical blued-steel verge pallet arbor and oscillating horizontal foliot balance beam with hanging poise weights.",
  },
  15: {
    id: 15,
    code: "arm.Jup",
    name: "Jupiter Carrier Arm & 4-Moon Galilean Epicycle",
    summary: "Fifth sleeve (z = 69.0, R = 76.0) driven at 14:58 carrying banded Jupiter (Great Red Spot) and its 4 Galilean moons (Io, Europa, Ganymede, Callisto) in Laplace resonance.",
  },
  16: {
    id: 16,
    code: "arm.Sat",
    name: "Saturn Carrier Arm, 26.73° Cassini Ring System & Titan",
    summary: "Outermost sixth coaxial sleeve (z = 65.0, R = 90.0) driven at 10:64 carrying banded Saturn, its 26.73° tilted concentric ring system (C, B, Cassini Division, A), and orbiting Titan.",
  },
};

// Right-handed 3D camera presets viewing from the front (-Y) hemisphere so -X is LEFT and +X is RIGHT!
export const CAMERA_PRESETS = {
  overview:   { label: "Overview",   az: -67,  el: 27.5, dist: 495, target: [-2, -2, 45],   fov: 29 },
  gearbox:    { label: "Gearbox",    az: -72,  el: 24,   dist: 124, target: [-11, -13, 23], fov: 31 },
  tellurian:  { label: "Tellurian",  az: -36,  el: 28,   dist: 76,  target: "track_earth",  fov: 32 },
  ephemeris:  { label: "Ephemeris",  az: -102, el: 27,   dist: 136, target: [90, -4, 28],   fov: 30 },
  davinci:    { label: "Da Vinci",   az: -64,  el: 27,   dist: 112, target: [-106, -5, 28], fov: 30 },
  ecliptic:   { label: "Plan View",  az: -90,  el: 84,   dist: 340, target: [-4, 0, 56],    fov: 29.5 },
};

// Material palettes (RGB albedo, materialId, roughness, metallic)
export const MATS = {
  brass:        { rgb: [0.85, 0.67, 0.30], id: 0,  rough: 0.25, metal: 0.88 },
  brassHi:      { rgb: [0.94, 0.78, 0.42], id: 0,  rough: 0.19, metal: 0.92 },
  brassLo:      { rgb: [0.68, 0.51, 0.21], id: 0,  rough: 0.36, metal: 0.82 },
  steel:        { rgb: [0.74, 0.77, 0.81], id: 1,  rough: 0.26, metal: 0.88 },
  chrome:       { rgb: [0.89, 0.91, 0.95], id: 2,  rough: 0.14, metal: 0.95 },
  copper:       { rgb: [0.84, 0.47, 0.30], id: 3,  rough: 0.27, metal: 0.86 },
  gunmetal:     { rgb: [0.22, 0.24, 0.27], id: 4,  rough: 0.40, metal: 0.68 },
  castIronDark: { rgb: [0.16, 0.17, 0.19], id: 4,  rough: 0.52, metal: 0.45 },
  castIronLight:{ rgb: [0.82, 0.80, 0.76], id: 4,  rough: 0.48, metal: 0.25 },
  sun:          { rgb: [1.00, 0.80, 0.32], id: 5,  rough: 0.20, metal: 0.20 },
  earth:        { rgb: [0.22, 0.46, 0.68], id: 6,  rough: 0.38, metal: 0.15 },
  mars:         { rgb: [0.78, 0.38, 0.22], id: 7,  rough: 0.55, metal: 0.10 },
  venus:        { rgb: [0.88, 0.76, 0.52], id: 8,  rough: 0.42, metal: 0.12 },
  mercury:      { rgb: [0.64, 0.62, 0.59], id: 9,  rough: 0.58, metal: 0.25 },
  moon:         { rgb: [0.76, 0.75, 0.72], id: 10, rough: 0.62, metal: 0.15 },
  dialDark:     { rgb: [0.15, 0.16, 0.18], id: 11, rough: 0.34, metal: 0.30 },
  dialLight:    { rgb: [0.95, 0.93, 0.88], id: 11, rough: 0.32, metal: 0.20 },
  ruby:         { rgb: [0.88, 0.14, 0.22], id: 12, rough: 0.14, metal: 0.15 },
  bluedSteel:   { rgb: [0.14, 0.28, 0.62], id: 13, rough: 0.20, metal: 0.90 },
  greekBronze:  { rgb: [0.62, 0.49, 0.28], id: 14, rough: 0.34, metal: 0.82 },
  jupiter:      { rgb: [0.84, 0.68, 0.48], id: 15, rough: 0.38, metal: 0.12 },
  saturn:       { rgb: [0.88, 0.76, 0.52], id: 16, rough: 0.34, metal: 0.14 },
  saturnRings:  { rgb: [0.90, 0.80, 0.60], id: 17, rough: 0.30, metal: 0.25 },
  astrolabe:    { rgb: [0.14, 0.16, 0.20], id: 18, rough: 0.32, metal: 0.45 },
};

// Fast 4x4 Column-Major Matrix Utilities (Zero Per-Frame Allocations)
export function mat4Identity(out) {
  out.fill(0);
  out[0] = out[5] = out[10] = out[15] = 1;
  return out;
}

export function mat4Multiply(out, a, b) {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  for (let i = 0; i < 4; i++) {
    const b0 = b[i * 4], b1 = b[i * 4 + 1], b2 = b[i * 4 + 2], b3 = b[i * 4 + 3];
    out[i * 4]     = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[i * 4 + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[i * 4 + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[i * 4 + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  }
  return out;
}

// Intrinsic-Z-first rotation R = Rx(rx) * Ry(ry) * Rz(rz):
// - rz spins the part around its own local +Z symmetry axis FIRST
// - (rx, ry) tilts/orients that local +Z axle into world direction [sin(ry), -sin(rx)*cos(ry), cos(rx)*cos(ry)]
// so spinning rz NEVER wobbles a tilted gear, lantern pinion, crown wheel, or planet globe!
export function composeTRS(out, tx, ty, tz, rx, ry, rz, sx, sy, sz) {
  const cx = Math.cos(rx), sx_ = Math.sin(rx);
  const cy = Math.cos(ry), sy_ = Math.sin(ry);
  const cz = Math.cos(rz), sz_ = Math.sin(rz);

  const r00 = cy * cz;
  const r10 = sx_ * sy_ * cz + cx * sz_;
  const r20 = -cx * sy_ * cz + sx_ * sz_;

  const r01 = -cy * sz_;
  const r11 = -sx_ * sy_ * sz_ + cx * cz;
  const r21 = cx * sy_ * sz_ + sx_ * cz;

  const r02 = sy_;
  const r12 = -sx_ * cy;
  const r22 = cx * cy;

  out[0] = r00 * sx; out[1] = r10 * sx; out[2] = r20 * sx; out[3] = 0;
  out[4] = r01 * sy; out[5] = r11 * sy; out[6] = r21 * sy; out[7] = 0;
  out[8] = r02 * sz; out[9] = r12 * sz; out[10] = r22 * sz; out[11] = 0;
  out[12] = tx;      out[13] = ty;      out[14] = tz;      out[15] = 1;

  const isx = 1 / (sx || 1), isy = 1 / (sy || 1), isz = 1 / (sz || 1);
  out[16] = r00 * isx; out[17] = r10 * isx; out[18] = r20 * isx; out[19] = 0;
  out[20] = r01 * isy; out[21] = r11 * isy; out[22] = r21 * isy; out[23] = 0;
  out[24] = r02 * isz; out[25] = r12 * isz; out[26] = r22 * isz; out[27] = 0;
  out[28] = 0;         out[29] = 0;         out[30] = 0;         out[31] = 1;
}

export function mat4LookAt(out, eye, target, up) {
  let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  let flen = Math.hypot(fx, fy, fz) || 1;
  fx /= flen; fy /= flen; fz /= flen;

  let sx = fy * up[2] - fz * up[1];
  let sy = fz * up[0] - fx * up[2];
  let sz = fx * up[1] - fy * up[0];
  let slen = Math.hypot(sx, sy, sz) || 1;
  sx /= slen; sy /= slen; sz /= slen;

  const ux = sy * fz - sz * fy;
  const uy = sz * fx - sx * fz;
  const uz = sx * fy - sy * fx;

  out[0] = sx; out[1] = ux; out[2] = -fx; out[3] = 0;
  out[4] = sy; out[5] = uy; out[6] = -fy; out[7] = 0;
  out[8] = sz; out[9] = uz; out[10] = -fz; out[11] = 0;
  out[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[14] = fz * eye[2] + fx * eye[0] + fy * eye[1];
  out[15] = 1;
  return out;
}

export function mat4PerspectiveZO(out, fovYRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovYRad * 0.5);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far / (near - far);
  out[11] = -1;
  out[14] = (far * near) / (near - far);
  return out;
}

export function mat4OrthoZO(out, left, right, bottom, top, near, far) {
  out.fill(0);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  out[10] = 1 / (near - far);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = near / (near - far);
  out[15] = 1;
  return out;
}

export function mat4Invert(out, m) {
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

  const b00 = m00 * m11 - m01 * m10;
  const b01 = m00 * m12 - m02 * m10;
  const b02 = m00 * m13 - m03 * m10;
  const b03 = m01 * m12 - m02 * m11;
  const b04 = m01 * m13 - m03 * m11;
  const b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30;
  const b07 = m20 * m32 - m22 * m30;
  const b08 = m20 * m33 - m23 * m30;
  const b09 = m21 * m32 - m22 * m31;
  const b10 = m21 * m33 - m23 * m31;
  const b11 = m22 * m33 - m23 * m32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return mat4Identity(out);
  det = 1.0 / det;

  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return out;
}

// Orbital & Horological Constants
export const T_EARTH = 365.25;
export const T_VENUS = 224.70;
export const T_MERC = 87.97;
export const T_MARS = 686.98;
export const T_JUP = 365.25 * (58 / 14); // Exact geared 14:58 (4.1429 yr = 1513.18 d) mechanical display rate
export const T_SAT = 365.25 * (64 / 10); // Exact geared 10:64 (6.4000 yr = 2337.60 d) mechanical display rate
export const T_MOON = 27.32166;
export const T_SUN_ROT = 25.38;

export function evaluateKinematics(day) {
  const dWrap = ((day % T_EARTH) + T_EARTH) % T_EARTH;
  const angE = (day / T_EARTH) * Math.PI * 2;
  const angV = (day / T_VENUS) * Math.PI * 2 + 0.85;
  const angM = (day / T_MERC) * Math.PI * 2 + 2.15;
  // Calibrated +1.92 rad epoch phase so exact Earth-Mars opposition (retrograde peak) occurs at Day 238.3 inside the [0, 365.25] slider!
  const angMars = (day / T_MARS) * Math.PI * 2 + 1.92;
  const angJup = (day / T_JUP) * Math.PI * 2 + 2.35;
  const angSat = (day / T_SAT) * Math.PI * 2 + 1.08;
  const titanAng = (day / 15.945) * Math.PI * 2 + 0.65;
  const angMoonMean = (day / T_MOON) * Math.PI * 2 + 0.45;
  const angNodal = -(day / (T_EARTH * 18.6)) * Math.PI * 2; // 18.6-year retrograde nodal precession
  const apsidalAng = (day / (T_EARTH * 8.85)) * Math.PI * 2; // 8.85-year prograde lunar apsidal precession (e5 carrier)

  // Antikythera Pin-and-Slot Hipparchus Lunar Anomaly (e5 / k1 / k2):
  // Eccentric drive pin at radius rPin = 5.0 on revolving apsidal center offset e = 0.58 modulates true lunar anomaly!
  const eccLocalX = 0.58 * Math.cos(apsidalAng);
  const eccLocalY = 0.58 * Math.sin(apsidalAng);
  const pinLocalX = eccLocalX + Math.cos(angMoonMean) * 5.0;
  const pinLocalY = eccLocalY + Math.sin(angMoonMean) * 5.0;
  const angMoon = Math.atan2(pinLocalY, pinLocalX);
  const pinSlotRadius = Math.hypot(pinLocalX, pinLocalY); // stays in [4.42, 5.58] inside slot [2.4, 7.6]!

  const lonDeg = Math.round(((angE * 180) / Math.PI % 360 + 360) % 360) % 360;
  const declDeg = 23.44 * Math.sin(angE);

  // Equation of Time (minutes): Δt = -7.65 sin(M) + 9.87 sin(2 λ_E)
  const meanAnom = angE - 0.049;
  const eotMin = -7.65 * Math.sin(meanAnom) + 9.87 * Math.sin(2 * angE);
  const kidneyCamRadius = 5.8 - 0.95 * Math.sin(angE - 0.049) + 1.25 * Math.sin(2 * angE);

  // Rigid-Linkage Equation-of-Time Bell-Crank Follower Solver (Law of Cosines):
  // Cam center C = [74.0, -3.0], Lever pivot P = [79.8, 10.0], Fixed roller arm L = 13.000, Ruby roller r = 0.95
  const rFollowerRoller = 0.95;
  const dcpX = 74.0 - 79.8; // -5.8
  const dcpY = -3.0 - 10.0; // -13.0
  const dcp = Math.hypot(dcpX, dcpY); // 14.2351677
  const baseAngPC = Math.atan2(dcpY, dcpX);
  const rCamPlusRoller = kidneyCamRadius + rFollowerRoller;
  const cosAlpha = (dcp * dcp + 13.0 * 13.0 - rCamPlusRoller * rCamPlusRoller) / (2.0 * dcp * 13.0);
  const alpha = Math.acos(Math.max(-1, Math.min(1, cosAlpha)));
  const eotLeverAngle = baseAngPC + alpha;
  const eotRollerX = 79.8 + 13.0 * Math.cos(eotLeverAngle);
  const eotRollerY = 10.0 + 13.0 * Math.sin(eotLeverAngle);

  // Piecewise Da Vinci Verge-and-Foliot Recoil & Impulse Law (20T Crown Wheel, 40 Pallet Beats/Rev):
  const crownTeeth = 20;
  const crownMeanRate = 8.0;
  const totalBeats = (day / T_EARTH) * crownMeanRate * crownTeeth * 2.0;
  const beatIdx = Math.floor(totalBeats);
  const beatPhase = ((totalBeats % 1) + 1) % 1;
  const wRecoil = 0.18;
  let beatDelta;
  let crownRecoilActive = false;
  if (beatPhase < wRecoil) {
    // Backward recoil phase: swinging foliot drives crown tooth backward (d(crownAngle)/dt < 0)
    beatDelta = -0.22 * Math.sin(((beatPhase / wRecoil) * Math.PI) * 0.5);
    crownRecoilActive = true;
  } else {
    // Forward impulse phase: crown tooth pushes pallet forward by +1.22 half-teeth (net +1.0 per beat)
    beatDelta = -0.22 + 1.22 * ((beatPhase - wRecoil) / (1.0 - wRecoil));
  }
  const crownAngle = ((beatIdx + beatDelta) * Math.PI) / crownTeeth;
  const palletSign = (((beatIdx % 2) + 2) % 2 === 0) ? 1 : -1;
  // C1-smooth foliot oscillation with exact turnaround extremum at beatPhase === wRecoil!
  const foliotAngle = 0.38 * palletSign * Math.cos(Math.PI * (beatPhase - wRecoil));

  // Maltese Cross (Geneva) 6-Slot Intermittent Indexing Law (300° Locked Dwell, 60° Vernal Step):
  // Center distance d_G = 7.20, drive pin radius r_p = 3.60 = 0.5 * d_G, wheel outer radius R_G = 6.2354
  const dGeneva = 7.20;
  const rPinGeneva = 3.60;
  const kYr = Math.floor((angE + Math.PI) / (Math.PI * 2));
  const alphaSolar = angE - kYr * Math.PI * 2; // in [-pi, pi) around Vernal Equinox alpha = 0
  let genevaAngle;
  let genevaDriverLocal;
  let genevaDwell = true;
  if (alphaSolar <= -Math.PI / 6) {
    genevaAngle = kYr * (Math.PI / 3);
    genevaDriverLocal = -Math.PI / 3 + ((alphaSolar + Math.PI / 6) * 4.0) / 5.0;
  } else if (alphaSolar >= Math.PI / 6) {
    genevaAngle = (kYr + 1) * (Math.PI / 3);
    genevaDriverLocal = Math.PI / 3 + ((alphaSolar - Math.PI / 6) * 4.0) / 5.0;
  } else {
    genevaDwell = false;
    genevaDriverLocal = 2.0 * alphaSolar; // maps [-pi/6, +pi/6] to [-pi/3, +pi/3] for exact C1 tangent entry/exit
    const deltaG = Math.atan2(
      rPinGeneva * Math.sin(genevaDriverLocal),
      dGeneva - rPinGeneva * Math.cos(genevaDriverLocal)
    );
    genevaAngle = (kYr + 0.5) * (Math.PI / 3) + deltaG;
  }
  const genevaDriverAngle = kYr * Math.PI * 2 + genevaDriverLocal;

  // Earth-Mars Geocentric Dioptra Sighting & Retrograde Apparent Motion:
  const R_E_ORB = 45.0, R_MARS_ORB = 62.0;
  const ex = -8.0 + R_E_ORB * Math.cos(angE);
  const ey = R_E_ORB * Math.sin(angE);
  const mx = -8.0 + R_MARS_ORB * Math.cos(angMars);
  const my = R_MARS_ORB * Math.sin(angMars);
  const dxEM = mx - ex, dyEM = my - ey;
  const distEM = Math.hypot(dxEM, dyEM) || 1;
  const marsGeoLon = Math.atan2(dyEM, dxEM);
  const omegaE = (Math.PI * 2) / T_EARTH;
  const omegaMars = (Math.PI * 2) / T_MARS;
  const vxE = -R_E_ORB * omegaE * Math.sin(angE), vyE = R_E_ORB * omegaE * Math.cos(angE);
  const vxM = -R_MARS_ORB * omegaMars * Math.sin(angMars), vyM = R_MARS_ORB * omegaMars * Math.cos(angMars);
  const marsGeoVel = (dxEM * (vyM - vyE) - dyEM * (vxM - vxE)) / (distEM * distEM);
  const marsRetrograde = marsGeoVel < 0;

  // Laplace 4:2:1 Galilean Resonance (lambda_Io - 3*lambda_Eu + 2*lambda_Ga === pi mod 2*pi):
  const omegaGa = 0.18;
  const galIoAng = 4.0 * omegaGa * day + 0.20;
  const galEuAng = 2.0 * omegaGa * day + 0.30;
  const galGaAng = 1.0 * omegaGa * day + 0.35 + Math.PI * 0.5;
  const galCaAng = (omegaGa * (3.0 / 7.0)) * day + 5.10;
  const laplacePhase = galIoAng - 3.0 * galEuAng + 2.0 * galGaAng; // === Math.PI (180.0°)

  // Syzygy eclipse alignment & Synodic lunar phase
  const relMoon = ((angMoon - angE) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
  const syzygyDiff = Math.min(
    Math.abs(relMoon),
    Math.abs(relMoon - Math.PI),
    Math.abs(relMoon - Math.PI * 2)
  );
  const eclipseStrength = Math.max(0, 1 - syzygyDiff / 0.28);

  let season = "Vernal";
  if (lonDeg >= 45 && lonDeg < 135) season = "Estival";
  else if (lonDeg >= 135 && lonDeg < 225) season = "Autumnal";
  else if (lonDeg >= 225 && lonDeg < 315) season = "Hibernal";

  const ZODIAC = ["Ari", "Tau", "Gem", "Cnc", "Leo", "Vir", "Lib", "Sco", "Sgr", "Cap", "Aqr", "Psc"];
  const zodiacSign = ZODIAC[Math.floor(lonDeg / 30) % 12];

  return {
    day,
    dWrap,
    angE,
    angV,
    angM,
    angMars,
    angJup,
    angSat,
    titanAng,
    angMoonMean,
    angMoon,
    pinSlotRadius,
    angNodal,
    apsidalAng,
    lonDeg,
    declDeg,
    eotMin,
    kidneyCamRadius,
    eotLeverAngle,
    eotRollerX,
    eotRollerY,
    crownAngle,
    crownRecoilActive,
    beatPhase,
    foliotAngle,
    genevaAngle,
    genevaDriverAngle,
    genevaDwell,
    marsGeoLon,
    marsGeoVel,
    marsRetrograde,
    galIoAng,
    galEuAng,
    galGaAng,
    galCaAng,
    laplacePhase,
    relMoon,
    eclipseStrength,
    season,
    zodiacSign,
  };
}

export function formatTelemetryForGroup(groupId, kin) {
  const info = GROUP_INFO[groupId];
  if (!info) return null;
  const deg = (r) => (((r * 180) / Math.PI % 360) + 360) % 360;
  let metrics = "";
  switch (groupId) {
    case 1:
      metrics = `Crank θ ${(deg(kin.angE * 4)).toFixed(0)}° · Worm 4.5-start : 28T · Fusee 24T Ratchet Click`;
      break;
    case 2:
      metrics = `Collinear y = -23.43 · Lantern 10-stave : Crown 20T · Bevel 18T : 18T · θ ${deg(kin.angE).toFixed(1)}°`;
      break;
    case 3:
      metrics = `Layshaft [-13.2, -23.43] · 62T / 48T / 40T / 28T / 14T / 10T · Pitch D = 24.00`;
      break;
    case 4:
      metrics = `6 Telescoping Tubes · Driven 16T / 30T / 40T / 52T / 58T / 64T · Zero Sleeve Overlap`;
      break;
    case 5:
      metrics = `Sidereal rot T = 25.38 d · φ_sun ${deg((kin.day / T_SUN_ROT) * Math.PI * 2).toFixed(0)}° · Point light z = 109.0`;
      break;
    case 6:
      metrics = `Ratio 62:16 (4.152×) · T = 87.97 d · R = 17.5 · i = 7.00° · λ_M ${deg(kin.angM).toFixed(0)}°`;
      break;
    case 7:
      metrics = `Ratio 48:30 (1.626×) · T = 224.70 d · R = 29.5 · i = 3.39° · λ_V ${deg(kin.angV).toFixed(0)}°`;
      break;
    case 8:
      metrics = `5-Gear Train + Vertical Jackshaft · Pin r = ${kin.pinSlotRadius.toFixed(2)} · Phase Φ_syn ${deg(kin.relMoon).toFixed(0)}°`;
      break;
    case 9:
      metrics = `Dioptra λ_geo ${deg(kin.marsGeoLon).toFixed(1)}° (${kin.marsRetrograde ? "RETROGRADE" : "PROGRADE"} ${(kin.marsGeoVel * 57.296).toFixed(2)}°/d) · λ_helio ${deg(kin.angMars).toFixed(0)}°`;
      break;
    case 10:
      metrics = `6 Fluted Corinthian Columns · Zodiac House ${kin.zodiacSign} (${kin.lonDeg}°) · Armillary Arch`;
      break;
    case 11:
      metrics = `Scotch-Yoke δ = ${kin.declDeg >= 0 ? "+" : ""}${kin.declDeg.toFixed(2)}° · Rigid Lever L = 13.00 · Geneva ${kin.genevaDwell ? "LOCKED" : "STEPPING"}`;
      break;
    case 12:
      metrics = `Day ${Math.round(kin.dWrap)} / 365 · Δt = ${kin.eotMin >= 0 ? "+" : ""}${kin.eotMin.toFixed(1)}m · Geneva θ_G ${deg(kin.genevaAngle).toFixed(0)}° · Ω ${deg(kin.angNodal).toFixed(1)}°`;
      break;
    case 13:
      metrics = `Leadscrew 22-turn · Traveling 14T Pinion (No-Slip) · Cursor x = ${(-114 + (kin.dWrap / T_EARTH) * 160).toFixed(1)}`;
      break;
    case 14:
      metrics = `Crown 20T (${kin.crownRecoilActive ? "RECOIL dθ<0" : "IMPULSE dθ>0"}) · Foliot ${(kin.foliotAngle * 57.3).toFixed(1)}°`;
      break;
    case 15:
      metrics = `4-Tier Coaxial Galilean Stack · Laplace Φ = ${deg(kin.laplacePhase).toFixed(1)}° (Io 4× : Eu 2× : Ga 1× : Ca)`;
      break;
    case 16:
      metrics = `Ratio 10:64 (6.40×) · R = 90.0 · Ring Obliquity 26.73° (Cassini Div) · λ_Sat ${deg(kin.angSat).toFixed(0)}° · Titan ${deg(kin.titanAng).toFixed(0)}°`;
      break;
  }
  return { ...info, metrics };
}

function populateOrreryScene(
  geometry,
  addPart,
  addSegmentRod,
  addPickSphere,
  pushOverlayVert,
  kin,
  eyePos,
  lightTheme,
  cutawayMode,
  explodeMode = 0
) {
  function addOrbitRibbon(cx, cy, cz, radius, width, incRad, ascNodeRad, rgb, alpha, segs = 96) {
    const cInc = Math.cos(incRad), sInc = Math.sin(incRad);
    const cNode = Math.cos(ascNodeRad), sNode = Math.sin(ascNodeRad);

    function ptOnOrbit(ang, r) {
      const xOrb = Math.cos(ang) * r;
      const yOrb = Math.sin(ang) * r;
      const x1 = xOrb;
      const y1 = yOrb * cInc;
      const z1 = yOrb * sInc;
      return [
        cx + x1 * cNode - y1 * sNode,
        cy + x1 * sNode + y1 * cNode,
        cz + z1,
      ];
    }

    const r0 = radius - width * 0.5;
    const r1 = radius + width * 0.5;
    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const p00 = ptOnOrbit(a0, r0);
      const p01 = ptOnOrbit(a0, r1);
      const p11 = ptOnOrbit(a1, r1);
      const p10 = ptOnOrbit(a1, r0);

      pushOverlayVert(p00[0], p00[1], p00[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 0, 0);
      pushOverlayVert(p01[0], p01[1], p01[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 1, 0);
      pushOverlayVert(p11[0], p11[1], p11[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 1, 0);

      pushOverlayVert(p00[0], p00[1], p00[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 0, 0);
      pushOverlayVert(p11[0], p11[1], p11[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 1, 0);
      pushOverlayVert(p10[0], p10[1], p10[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], alpha, 0, 0);
    }
  }

  // Tapered Keplerian Velocity Comet-Tail Arc trailing behind each orbiting planet
  function addVelocityTailArc(cx, cy, cz, radius, widthHead, curAng, spanRad, incRad, ascNodeRad, rgb, alphaHead, segs = 28) {
    const cInc = Math.cos(incRad), sInc = Math.sin(incRad);
    const cNode = Math.cos(ascNodeRad), sNode = Math.sin(ascNodeRad);
    function ptOnOrbit(ang, r) {
      const xOrb = Math.cos(ang) * r;
      const yOrb = Math.sin(ang) * r;
      const x1 = xOrb;
      const y1 = yOrb * cInc;
      const z1 = yOrb * sInc;
      return [
        cx + x1 * cNode - y1 * sNode,
        cy + x1 * sNode + y1 * cNode,
        cz + z1 + 0.08,
      ];
    }
    for (let i = 0; i < segs; i++) {
      const t0 = i / segs;
      const t1 = (i + 1) / segs;
      const a0 = curAng - spanRad * (1 - t0);
      const a1 = curAng - spanRad * (1 - t1);
      const w0 = widthHead * (0.15 + 0.85 * t0);
      const w1 = widthHead * (0.15 + 0.85 * t1);
      const al0 = alphaHead * t0 * t0;
      const al1 = alphaHead * t1 * t1;
      const p00 = ptOnOrbit(a0, radius - w0 * 0.5);
      const p01 = ptOnOrbit(a0, radius + w0 * 0.5);
      const p11 = ptOnOrbit(a1, radius + w1 * 0.5);
      const p10 = ptOnOrbit(a1, radius - w1 * 0.5);

      pushOverlayVert(p00[0], p00[1], p00[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al0, 0, 0);
      pushOverlayVert(p01[0], p01[1], p01[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al0, 1, 0);
      pushOverlayVert(p11[0], p11[1], p11[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al1, 1, 0);

      pushOverlayVert(p00[0], p00[1], p00[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al0, 0, 0);
      pushOverlayVert(p11[0], p11[1], p11[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al1, 1, 0);
      pushOverlayVert(p10[0], p10[1], p10[2], 0, 0, 1, rgb[0], rgb[1], rgb[2], al1, 0, 0);
    }
  }

  function addConeFrustumOverlay(pStart, rStart, pEnd, rEnd, rgb, alpha, segs = 28) {
    const dx = pEnd[0] - pStart[0], dy = pEnd[1] - pStart[1], dz = pEnd[2] - pStart[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    const ax = dx / len, ay = dy / len, az = dz / len;
    let ux = -ay, uy = ax, uz = 0;
    let ulen = Math.hypot(ux, uy, uz);
    if (ulen < 0.01) { ux = 1; uy = 0; uz = 0; ulen = 1; }
    ux /= ulen; uy /= ulen; uz /= ulen;
    const vx = ay * uz - az * uy;
    const vy = az * ux - ax * uz;
    const vz = ax * uy - ay * ux;

    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2;
      const a1 = ((i + 1) / segs) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      const n0x = ux * c0 + vx * s0, n0y = uy * c0 + vy * s0, n0z = uz * c0 + vz * s0;
      const n1x = ux * c1 + vx * s1, n1y = uy * c1 + vy * s1, n1z = uz * c1 + vz * s1;

      const p00 = [pStart[0] + n0x * rStart, pStart[1] + n0y * rStart, pStart[2] + n0z * rStart];
      const p10 = [pStart[0] + n1x * rStart, pStart[1] + n1y * rStart, pStart[2] + n1z * rStart];
      const p11 = [pEnd[0] + n1x * rEnd, pEnd[1] + n1y * rEnd, pEnd[2] + n1z * rEnd];
      const p01 = [pEnd[0] + n0x * rEnd, pEnd[1] + n0y * rEnd, pEnd[2] + n0z * rEnd];

      pushOverlayVert(p00[0], p00[1], p00[2], n0x, n0y, n0z, rgb[0], rgb[1], rgb[2], alpha, 0.05, 2.0);
      pushOverlayVert(p10[0], p10[1], p10[2], n1x, n1y, n1z, rgb[0], rgb[1], rgb[2], alpha, 0.05, 2.0);
      pushOverlayVert(p11[0], p11[1], p11[2], n1x, n1y, n1z, rgb[0], rgb[1], rgb[2], alpha * 0.25, 0.95, 2.0);

      pushOverlayVert(p00[0], p00[1], p00[2], n0x, n0y, n0z, rgb[0], rgb[1], rgb[2], alpha, 0.05, 2.0);
      pushOverlayVert(p11[0], p11[1], p11[2], n1x, n1y, n1z, rgb[0], rgb[1], rgb[2], alpha * 0.25, 0.95, 2.0);
      pushOverlayVert(p01[0], p01[1], p01[2], n0x, n0y, n0z, rgb[0], rgb[1], rgb[2], alpha * 0.25, 0.95, 2.0);
    }
  }

  // World Coordinates (Right-Handed: -X = Left, +X = Right, -Y = Front, +Y = Back, +Z = Up)
  const CENTER_XY = [-8.0, 0.0];
  const LAY_XY = [-13.2, -23.43]; // Front-left cutaway window! Distance D = hypot(-5.2, -23.43) = 24.00 exact!
  const CRANK_XY = [-121.0, 11.0];
  const FUSEE_XY = [-125.0, -15.5];
  const MBARREL_XY = [-104.0, -15.5];
  const ESCAPEMENT_XY = [-84.0, -11.0];
  const DECL_XY = [95.2, -11.8];
  const CAM_XY = [74.0, -3.0];
  const DIAL_XY = [104.0, 12.0];
  const ECLIPTIC_PLANE_Z = 109.0;
  const SUN_POS = [CENTER_XY[0], CENTER_XY[1], ECLIPTIC_PLANE_Z];
  const SUN_R = 6.8;

  let earthPos = [-8 + 45, 0, ECLIPTIC_PLANE_Z];
  let moonPos = [-8 + 54, 0, ECLIPTIC_PLANE_Z];

  const bedMat = lightTheme ? MATS.castIronLight : MATS.castIronDark;
  const dialFaceMat = lightTheme ? MATS.dialLight : MATS.dialDark;

  // ========================================================================
  // GROUP 13: Foundation Bedplate, Bun Feet, Screws & Toothed Linear Ephemeris Rack
  // ========================================================================
  addPart("bedplate", 0, 0, 0, 0, 0, 0, 1, 1, 1, bedMat, 13);
  addPart("bedInlay", 0, 0, 0, 0, 0, 0, 1, 1, 1, MATS.gunmetal, 13);
  for (const [fx, fy] of [
    [-134, -70], [134, -70], [-134, 70], [134, 70], [0, -70], [0, 70],
  ]) {
    addPart("unitCyl", fx, fy, -14.8, 0, 0, 0, 8.4, 8.4, 1.8, MATS.brassLo, 13);
    addPart("unitCyl", fx, fy, -12.0, 0, 0, 0, 6.2, 6.2, 4.0, MATS.brass, 13);
  }
  for (const [sx, sy] of [
    [-144, -80], [144, -80], [-144, 80], [144, 80], [0, -82], [0, 82],
  ]) {
    addPart("unitCyl", sx, sy, 0.9, 0, 0, 0, 2.3, 2.3, 0.5, MATS.brassHi, 13);
    addPart("slottedScrew", sx, sy, 1.15, 0, 0, 0.4, 1.35, 1.35, 1.0, MATS.bluedSteel, 13);
  }

  // Front Toothed Linear Ephemeris Rack (y = -74.0, facing viewer) & Sliding Vernier Microscope
  const rackY = -74.0;
  addPart("rulerStrip", -34, rackY, 0.8, 0, 0, 0, 1, 1, 1, MATS.brass, 13);
  // Helical Acme Leadscrew carrying the sliding Vernier carriage across [-121, +53] at y = rackY - 1.6 = -75.6, z = 3.25
  // (rTip = 0.68 -> zMin = 2.57 > graduation ticks zMax = 2.52 and base z = 2.20, zero rack collision!)
  addPart("rackLeadscrew", -34.0, rackY - 1.6, 3.25, 0, Math.PI * 0.5, kin.angE * 16.0, 1, 1, 1, MATS.steel, 13);
  addPart("unitCyl", -121.5, rackY - 1.6, 2.2, 0, 0, 0, 1.35, 1.35, 2.8, MATS.brassLo, 13);
  addPart("unitCyl", 53.5, rackY - 1.6, 2.2, 0, 0, 0, 1.35, 1.35, 2.8, MATS.brassLo, 13);

  const cursorX = -114 + (kin.dWrap / T_EARTH) * 160;
  const rackPinionRot = -(cursorX - (-114.0)) / 2.80;
  addPart("transferBox", cursorX, rackY - 0.8, 2.15, 0, 0, 0, 0.58, 0.72, 0.28, MATS.chrome, 13);
  // Coaxial threaded brass leadscrew nut glands flanking the Vernier carriage along x = cursorX +- 3.45
  addPart("unitCyl", cursorX - 3.45, rackY - 1.6, 3.25, 0, Math.PI * 0.5, 0, 0.88, 0.88, 0.45, MATS.brassHi, 13);
  addPart("unitCyl", cursorX + 3.45, rackY - 1.6, 3.25, 0, Math.PI * 0.5, 0, 0.88, 0.88, 0.45, MATS.brassHi, 13);
  // Traveling 14T rack pinion mounted on the Vernier carriage at z = 4.95 (rPitch = 2.80 -> bottom pitch z = 2.15 engages rack teeth with zero slip!)
  addSegmentRod([cursorX, rackY - 1.6, 4.95], [cursorX, rackY + 3.0, 4.95], 0.48, MATS.steel, 13);
  addPart("gear_rackPinion", cursorX, rackY + 2.2, 4.95, Math.PI * 0.5, 0, rackPinionRot, 1, 1, 1, MATS.copper, 13);
  // Coaxial Vernier Fractional-Day Micro-Dial & Pointer Needle on top of traveling carriage
  addPart("unitCyl", cursorX, rackY - 0.8, 4.65, 0, 0, 0, 2.2, 2.2, 1.1, MATS.brassHi, 13);
  addPart("dialFace", cursorX, rackY - 0.8, 5.15, 0, 0, 0, 0.098, 0.098, 0.35, dialFaceMat, 13);
  addPart("hand_sun", cursorX, rackY - 0.8, 5.56, 0, 0, rackPinionRot * 4.0, 0.105, 0.105, 0.38, MATS.bluedSteel, 13);
  addPart("unitCyl", cursorX, rackY - 0.8, 5.72, 0, 0, 0, 0.36, 0.36, 0.24, MATS.ruby, 13);
  addPickSphere(cursorX, rackY, 3.0, 14, 13);
  addPickSphere(-55, rackY, 2.0, 18, 13);

  // Maker's Plaque on front right (y = -74.0) with 4 corner escutcheon bosses, blued screws & Calibre medallion
  addPart("makerPlaque", 102, rackY, 0.8, 0, 0, 0, 1, 1, 1, MATS.brassLo, 13);
  for (const [mxOff, myOff] of [[-34.2, -3.1], [34.2, -3.1], [-34.2, 3.1], [34.2, 3.1]]) {
    addPart("unitCyl", 102 + mxOff, rackY + myOff, 1.80, 0, 0, 0, 0.92, 0.92, 0.28, MATS.brassHi, 13);
    addPart("slottedScrew", 102 + mxOff, rackY + myOff, 1.96, 0, 0, mxOff * 0.1, 0.56, 0.56, 0.65, MATS.bluedSteel, 13);
  }
  addPart("jewelChaton", 102, rackY, 1.82, 0, 0, 0, 0.85, 0.85, 0.55, MATS.brassHi, 13);
  addPart("unitCyl", 102, rackY, 1.88, 0, 0, 0, 0.92, 0.92, 0.35, MATS.ruby, 13);

  // ========================================================================
  // GROUP 1: Left Assembly — Handwheel Crank, Cutaway Worm Barrel & Da Vinci Conical Fusee
  // ========================================================================
  const [kx, ky] = CRANK_XY;
  addPart("driveBox", -108, 0, 0, 0, 0, 0, 1, 1, 1, MATS.gunmetal, 1);
  for (const [bx, by, zSurf] of [
    [-138.4, -28.4, 14.0], [-108.0, -28.4, 14.0], [-77.6, -28.4, 8.5],
    [-138.4,  28.4, 14.0], [-108.0,  28.4, 14.0], [-77.6,  28.4, 14.0],
  ]) {
    addPart("unitCyl", bx, by, zSurf + 0.15, 0, 0, 0, 1.55, 1.55, 0.35, MATS.brassHi, 1);
    addPart("unitCyl", bx, by, zSurf + 0.42, 0, 0, 0, 1.0, 1.0, 0.45, MATS.bluedSteel, 1);
    addPart("daVinciWingNut", bx, by, zSurf + 0.58, 0, 0, 0.4, 0.62, 0.62, 0.62, MATS.brass, 1);
  }

  // Cutaway Worm-Gear Barrel (window facing -Y front!) showing internal helical worm & 28T worm wheel
  addPart("unitCyl", kx, ky, 16.75, 0, 0, 0, 13.4, 13.4, 0.9, MATS.brass, 1);
  addPart("wormBarrel", kx, ky, 17.2, 0, 0, 0, 1, 1, 1, MATS.brass, 1, 1);
  addPart("unitCyl", kx, ky, 37.3, 0, 0, 0, 11.4, 11.4, 3.2, MATS.brass, 1);
  addPart("unitCyl", kx, ky, 40.8, 0, 0, 0, 5.4, 5.4, 3.8, MATS.brassLo, 1);
  // Graduated 12-division worm-turns index bezel & blued-steel pointer on upper gland
  addPart("dialFace", kx, ky, 42.75, 0, 0, 0, 0.27, 0.27, 0.35, dialFaceMat, 1);
  addPart("oilCup", kx + 3.4, ky, 42.6, 0, 0, 0, 0.82, 0.82, 0.82, MATS.brassHi, 1);

  // 4 Vertical External Tie-Rod Columns & Da Vinci Butterfly Wing-Nuts clamping the cutaway worm barrel
  for (const tieAng of [0.35, 1.25, 1.90, 2.80]) {
    const tx = kx + Math.cos(tieAng) * 11.8;
    const ty = ky + Math.sin(tieAng) * 11.8;
    addPart("unitCyl", tx, ty, 28.0, 0, 0, 0, 0.52, 0.52, 22.0, MATS.steel, 1);
    addPart("unitCyl", tx, ty, 17.4, 0, 0, 0, 1.15, 1.15, 0.6, MATS.brassLo, 1);
    addPart("daVinciWingNut", tx, ty, 38.85, 0, 0, tieAng + Math.PI * 0.5, 0.72, 0.72, 0.72, MATS.brassHi, 1);
  }
  // 8 Base Flange Hold-Down Studs around wormBarrel base ring
  for (let b = 0; b < 8; b++) {
    const ba = (b / 8) * Math.PI * 2 + 0.2;
    addPart("unitCyl", kx + Math.cos(ba) * 12.1, ky + Math.sin(ba) * 12.1, 17.35, 0, 0, 0, 0.55, 0.55, 0.55, MATS.bluedSteel, 1);
  }

  // Da Vinci Conical Helical Fusee Drum ("Cono Elicoidale") & Mainspring Barrel on Front Deck of Drive Box
  const [fx, fy] = FUSEE_XY;
  const [mbx, mby] = MBARREL_XY;
  const rMainspring = 5.8;

  // Taut Fusee Chain / Cord tangent span tracking the exact hyperboloid profile radAt(tFusee) of daVinciFusee!
  const chainZ = 17.8 + (kin.dWrap / T_EARTH) * 10.5;
  const tFusee = (chainZ - 16.2) / 13.5;
  const rFuseeAtZ = 6.4 * (1 - tFusee) + 3.2 * tFusee - 0.55 * Math.sin(tFusee * Math.PI) + 0.22;
  // Exact front-facing external tangent normal perpendicular to [fx, fy] -> [mbx, mby] (thetaTan = -pi/2 since fy === mby = -15.5)
  const thetaTan = Math.atan2(mby - fy, mbx - fx) - Math.PI * 0.5;
  // Because buildConicalFusee winds its 6-turn helical groove clockwise (-t * 12*pi), locking fuseeRot = thetaTan + tFusee * 12*pi
  // places the groove at thetaTan (-pi/2) at height chainZ, and as t increases up the cone, angle aW = fuseeRot - t * 12*pi DECREASES
  // (continuing in the -X direction around the front-left of the cone in 100% C^1 alignment with the arriving chain from the Mainspring Barrel!)
  const fuseeRot = thetaTan + tFusee * 6.0 * Math.PI * 2;
  const barrelRot = fuseeRot * 0.72;
  const windDropRot = fuseeRot * (7.14 / (6.8 * 0.5846));
  const crankAngle = kin.angE * 4.0;

  addPart("wormScrew", kx, ky - 1.5, 26.4, 0, 0, crankAngle * 7.0, 1, 1, 1, MATS.steel, 1);
  // Upper & lower thrust bearing collars on the vertical worm screw arbor
  addPart("unitCyl", kx, ky - 1.5, 19.2, 0, 0, 0, 2.3, 2.3, 1.1, MATS.brassLo, 1);
  addPart("unitCyl", kx, ky - 1.5, 33.8, 0, 0, 0, 2.3, 2.3, 1.1, MATS.brassLo, 1);
  // Worm wheel spins around its horizontal +Y axle (rx = Math.PI * 0.5, ry = 0, rz = -windDropRot)
  addPart("gear_wormWheel", kx + 4.4, ky - 2.4, 25.5, Math.PI * 0.5, 0, -windDropRot, 0.85, 0.85, 0.85, MATS.copper, 1);

  // Input spindle & 4-spoke knurled handwheel at z = 49..54
  addPart("unitCyl", kx, ky, 46.5, 0, 0, crankAngle, 2.6, 2.6, 11.0, MATS.steel, 1);
  addPart("handwheelRim", kx, ky, 50.5, 0, 0, crankAngle, 1, 1, 1, MATS.brassHi, 1);
  addPart("unitCyl", kx, ky, 50.5, 0, 0, crankAngle, 4.0, 4.0, 4.8, MATS.brass, 1);
  for (let s = 0; s < 4; s++) {
    const sa = crankAngle + (s * Math.PI) / 2;
    const p0 = [kx + Math.cos(sa) * 3.4, ky + Math.sin(sa) * 3.4, 50.5];
    const p1 = [kx + Math.cos(sa) * 11.2, ky + Math.sin(sa) * 11.2, 50.5];
    addSegmentRod(p0, p1, 0.95, MATS.brass, 1);
    // Turned spoke root & rim ferrules
    addPart("unitCyl", kx + Math.cos(sa) * 4.2, ky + Math.sin(sa) * 4.2, 50.5, 0, 0, sa, 1.25, 1.25, 2.1, MATS.brassLo, 1);
  }
  addPart("unitCyl", kx, ky, 54.0, 0, 0, crankAngle, 3.6, 3.6, 2.2, MATS.bluedSteel, 1);
  addPart("daVinciWingNut", kx, ky, 55.05, 0, 0, crankAngle, 1.15, 1.15, 1.1, MATS.brassHi, 1);

  // Eccentric rotating crank handle pin on handwheel rim with revolving grip & brass ferrules
  const hx = kx + Math.cos(crankAngle) * 12.2;
  const hy = ky + Math.sin(crankAngle) * 12.2;
  addPart("unitCyl", hx, hy, 53.6, 0, 0, 0, 2.4, 2.4, 1.2, MATS.brassLo, 1);
  addPart("unitCyl", hx, hy, 56.2, 0, 0, 0, 1.9, 1.9, 6.4, MATS.steel, 1);
  addPart("balusterPillar", hx, hy, 59.8, 0, 0, 0, 1.25, 1.25, 0.95, MATS.copper, 1);
  addPart("unitCyl", hx, hy, 61.4, 0, 0, 0, 2.8, 2.8, 4.2, MATS.brassHi, 1);
  addPart("unitCyl", hx, hy, 63.8, 0, 0, 0, 1.35, 1.35, 0.85, MATS.ruby, 1);

  addPart("gear_wormWheel", fx, fy, 15.2, 0, 0, fuseeRot, 1.05, 1.05, 0.9, MATS.copper, 1);
  addPart("daVinciFusee", fx, fy, 16.2, 0, 0, fuseeRot, 1, 1, 1, MATS.brassHi, 1);
  addPart("unitCyl", fx, fy, 31.2, 0, 0, fuseeRot, 1.65, 1.65, 3.2, MATS.bluedSteel, 1);
  addPart("unitCyl", fx, fy, 32.8, 0, 0, fuseeRot, 0.88, 0.88, 3.6, MATS.steel, 1);
  // Smooth 36-segment S-scroll Bridge Cock ("Coq") anchored on the outboard left Drive Box deck (rz = 0, pillar at [-135.6, -15.5])
  // so it is seen in full broadside profile while keeping the front (-Y) chain and helical groove 100% unobstructed!
  const fuseeCockAng = 0.0;
  addPart("horologicalCock", fx, fy, 30.2, 0, 0, fuseeCockAng, 1.0, 1.0, 1.0, MATS.brassHi, 1);
  addPart("jewelChaton", fx, fy, 34.45, 0, 0, 0, 1.15, 1.15, 0.9, MATS.brassHi, 1);
  addPart("slottedScrew", fx - 10.6, fy, 31.55, 0, 0, 0.35, 0.85, 0.85, 0.95, MATS.bluedSteel, 1);
  addPart(
    "unitCyl",
    fx + Math.cos(fuseeCockAng) * -10.6,
    fy + Math.sin(fuseeCockAng) * -10.6,
    22.8,
    0,
    0,
    0,
    1.25,
    1.25,
    17.6,
    MATS.brassLo,
    1
  );
  addPart("unitCyl", fx, fy, 34.55, 0, 0, 0, 0.95, 0.95, 0.45, MATS.ruby, 1);

  // Openworked Swiss Skeleton Mainspring Barrel ("Tambour Squelette", Patek Philippe Calatrava Cross style) at [mbx, mby]
  // Windowed 3-ring skeleton barrel cage with 4 vertical ribs revealing the 7-turn coiled blued-steel mainspring leaves in 3D profile
  addPart("unitTube", mbx, mby, 17.0, 0, 0, barrelRot, rMainspring, rMainspring, 1.1, MATS.brass, 1);
  addPart("unitTube", mbx, mby, 21.6, 0, 0, barrelRot, rMainspring, rMainspring, 0.55, MATS.brassHi, 1);
  addPart("unitTube", mbx, mby, 26.3, 0, 0, barrelRot, rMainspring, rMainspring, 1.15, MATS.brass, 1);
  addPart("unitCyl", mbx, mby, 16.55, 0, 0, barrelRot, rMainspring + 0.42, rMainspring + 0.42, 0.55, MATS.brassHi, 1);
  addPart("unitTube", mbx, mby, 27.0, 0, 0, barrelRot, rMainspring + 0.42, rMainspring + 0.42, 0.68, MATS.brassHi, 1);
  // Central blued-steel barrel arbor + 5 stacked 7-turn coiled blued-steel mainspring ribbons with crisp radial coil gaps
  addPart("unitCyl", mbx, mby, 21.8, 0, 0, fuseeRot, 1.12, 1.12, 11.0, MATS.bluedSteel, 1);
  for (const szSpring of [17.8, 19.7, 21.6, 23.5, 25.4]) {
    addPart("hairspring", mbx, mby, szSpring, 0, 0, barrelRot, 1.18, 1.18, 1.05, MATS.bluedSteel, 1);
  }
  // 4 vertical skeleton cage pillars & Patek Philippe Calatrava Cross openworked top barrel cover spanning the full r = 5.8 barrel drum
  for (let k = 0; k < 4; k++) {
    const spAng = barrelRot + (k * Math.PI) / 2;
    const cs = Math.cos(spAng), sn = Math.sin(spAng);
    const px = -sn, py = cs;
    addPart("unitCyl", mbx + cs * (rMainspring - 0.22), mby + sn * (rMainspring - 0.22), 21.75, 0, 0, spAng, 0.34, 0.34, 9.9, MATS.brassHi, 1);
    // Calatrava cross radial arm spanning r = 1.25..5.65 (intrinsic +Z aligned radially along [cs, sn, 0]!) + flared fleur-de-lis crossbars at r = 4.45
    const armRx = Math.atan2(-sn, 0);
    const armRy = Math.asin(Math.max(-1, Math.min(1, cs)));
    addPart("unitCyl", mbx + cs * 3.45, mby + sn * 3.45, 27.06, armRx, armRy, 0, 0.36, 0.36, 4.4, MATS.brassHi, 1);
    addPart("unitCyl", mbx + cs * 4.45 + px * 0.78, mby + sn * 4.45 + py * 0.78, 27.06, 0, 0, spAng, 0.56, 0.30, 0.46, MATS.brassHi, 1);
    addPart("unitCyl", mbx + cs * 4.45 - px * 0.78, mby + sn * 4.45 - py * 0.78, 27.06, 0, 0, spAng, 0.56, 0.30, 0.46, MATS.brassHi, 1);
  }
  // Compact central power-reserve medallion (r = 1.38) at center of the Calatrava Cross so all 4 open quadrants expose the 7-turn spiral coils!
  addPart("unitCyl", mbx, mby, 27.65, 0, 0, 0, 1.38, 1.38, 0.68, MATS.brassHi, 1);
  addPart("dialFace", mbx, mby, 27.95, 0, 0, 0, 0.072, 0.072, 0.32, dialFaceMat, 1);
  addPart("jewelChaton", mbx, mby, 28.22, 0, 0, 0, 0.24, 0.24, 0.35, MATS.brassHi, 1);
  addPart("hand_eot", mbx, mby, 28.32, 0, 0, (kin.dWrap / T_EARTH) * Math.PI * 1.6 - 0.8, 0.082, 0.082, 0.36, MATS.bluedSteel, 1);
  addPart("unitCyl", mbx, mby, 28.44, 0, 0, 0, 0.24, 0.24, 0.22, MATS.ruby, 1);

  const cosTan = Math.cos(thetaTan);
  const sinTan = Math.sin(thetaTan);
  const pChainBarrel = [mbx + cosTan * rMainspring, mby + sinTan * rMainspring, chainZ];
  const pChainFusee = [fx + cosTan * rFuseeAtZ, fy + sinTan * rFuseeAtZ, chainZ];
  addSegmentRod(pChainBarrel, pChainFusee, 0.24, MATS.steel, 1);

  // Multi-Link Articulated Da Vinci Roller Chain ("Catena Articolata", Codex Madrid I f. 10r):
  // 1) 15 articulated flat-plate chain links spanning the exact C^1 tangent run from Mainspring Barrel (u=0) to Conical Fusee (u=1)
  const chainYaw = Math.atan2(pChainFusee[1] - pChainBarrel[1], pChainFusee[0] - pChainBarrel[0]);
  const chainLinksSpan = 15;
  for (let c = 0; c < chainLinksSpan; c++) {
    const u = c / (chainLinksSpan - 1);
    const lx = pChainBarrel[0] + (pChainFusee[0] - pChainBarrel[0]) * u;
    const ly = pChainBarrel[1] + (pChainFusee[1] - pChainBarrel[1]) * u;
    const scaleZ = c % 2 === 0 ? 0.88 : 0.74;
    addPart("fuseeChainLink", lx, ly, chainZ, 0, 0, chainYaw, 0.98, 0.9, scaleZ, c % 2 === 0 ? MATS.steel : MATS.brassHi, 1);
  }
  // 2) Continuous 3D steel chain core cable seated inside the Fusee helical groove from pChainFusee (t = tFusee) up to the top anchor (t = 0.96)
  const helixCableSegs = 64;
  for (let s = 0; s < helixCableSegs; s++) {
    const t0 = tFusee + (s / helixCableSegs) * (0.96 - tFusee);
    const t1 = tFusee + ((s + 1) / helixCableSegs) * (0.96 - tFusee);
    const r0 = 6.4 * (1 - t0) + 3.2 * t0 - 0.55 * Math.sin(t0 * Math.PI) + 0.12;
    const r1 = 6.4 * (1 - t1) + 3.2 * t1 - 0.55 * Math.sin(t1 * Math.PI) + 0.12;
    const a0 = fuseeRot - t0 * 6.0 * Math.PI * 2;
    const a1 = fuseeRot - t1 * 6.0 * Math.PI * 2;
    addSegmentRod(
      [fx + Math.cos(a0) * r0, fy + Math.sin(a0) * r0, 16.2 + t0 * 13.5],
      [fx + Math.cos(a1) * r1, fy + Math.sin(a1) * r1, 16.2 + t1 * 13.5],
      0.22,
      MATS.steel,
      1
    );
  }
  // 3) 96 contiguous pin-to-pin chain links filling 100% of the Fusee helical groove from pChainFusee (t = tFusee) up to the top anchor (t = 0.96):
  const fuseeHelixLinks = 96;
  for (let wIdx = 0; wIdx < fuseeHelixLinks; wIdx++) {
    const tW = tFusee + ((wIdx + 1) / fuseeHelixLinks) * (0.96 - tFusee);
    const zW = 16.2 + tW * 13.5;
    const rW = 6.4 * (1 - tW) + 3.2 * tW - 0.55 * Math.sin(tW * Math.PI) + 0.14;
    const aW = fuseeRot - tW * 6.0 * Math.PI * 2;
    addPart(
      "fuseeChainLink",
      fx + Math.cos(aW) * rW,
      fy + Math.sin(aW) * rW,
      zW,
      0.085,
      0,
      aW - Math.PI * 0.5,
      0.92,
      0.84,
      0.78,
      wIdx % 2 === 0 ? MATS.steel : MATS.brassHi,
      1
    );
  }
  // 4) Full 360-degree wrap (25 contiguous links) seated on a turned brass keeper flange around the Mainspring Barrel:
  addPart("unitCyl", mbx, mby, chainZ - 0.34, 0, 0, 0, rMainspring + 0.42, rMainspring + 0.42, 0.24, MATS.brassHi, 1);
  const barrelWrapLinks = 25;
  for (let bIdx = 0; bIdx < barrelWrapLinks; bIdx++) {
    const bAng = thetaTan + (bIdx / barrelWrapLinks) * Math.PI * 2;
    addPart(
      "fuseeChainLink",
      mbx + Math.cos(bAng) * (rMainspring + 0.14),
      mby + Math.sin(bAng) * (rMainspring + 0.14),
      chainZ,
      0,
      0,
      bAng - Math.PI * 0.5,
      0.92,
      0.84,
      0.8,
      bIdx % 2 === 0 ? MATS.steel : MATS.brassHi,
      1
    );
  }

  // Closed-Loop Outboard Winding & Fusee Ratchet Click Train:
  // 1) Horizontal worm-wheel output axle along Y at [kx + 4.4 = -116.6, z = 25.5] exiting through a side-wall journal boss to y = -3.6
  // (strictly outside the r = 13.4 Worm Barrel base flange so the cutaway worm-screw window is 100% uncluttered and no shaft impales the flange!)
  const dropX = kx + 4.4; // -116.6
  const dropY = -3.6;     // distance from [-121, 11] is hypot(4.4, -14.6) = 15.25 > 13.4!
  addSegmentRod([dropX, dropY + 1.4, 25.5], [dropX, ky + 2.2, 25.5], 0.82, MATS.steel, 1);
  addPart("unitCyl", dropX, -0.8, 25.5, Math.PI * 0.5, 0, 0, 1.65, 1.65, 1.4, MATS.brassLo, 1);
  addPart("unitCyl", dropX, ky + 1.8, 19.8, 0, 0, 0, 1.25, 1.25, 11.4, MATS.brassLo, 1);
  addPart("oilCup", dropX, -0.8, 27.1, 0, 0, 0, 0.72, 0.72, 0.72, MATS.brassHi, 1);
  // Outboard cast-brass bearing pillar at [dropX, -0.8] supporting the 90° miter-bevel drop arbor above z = 16.5
  addPart("unitCyl", dropX, -0.8, 21.0, 0, 0, 0, 0.55, 0.55, 9.0, MATS.brassLo, 1);
  addPart("unitCyl", dropX, dropY - 0.65, 25.5, 0, 0, 0, 1.45, 1.85, 0.85, MATS.brassHi, 1);
  addPart("oilCup", dropX, dropY - 1.35, 25.55, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 1);
  addPart("bevelGear", dropX, dropY + 1.4, 25.5, Math.PI * 0.5, 0, -windDropRot, 0.55, 0.55, 0.55, MATS.brassHi, 1);
  addPart("bevelGear", dropX, dropY, 24.1, 0, 0, windDropRot, 0.55, 0.55, 0.55, MATS.copper, 1);
  addPart("unitCyl", dropX, dropY, 19.65, 0, 0, windDropRot, 0.78, 0.78, 8.9, MATS.steel, 1);
  addPart("gear_wormWheel", dropX, dropY, 15.2, 0, 0, windDropRot, 0.5846, 0.5846, 0.65, MATS.brassHi, 1);
  // Intermediate winding pinion at [-121.8, -6.8, 15.2] (rPitch = 2.10 * 1.0143 = 2.130) tangent to drop gear (3.975+2.130=6.105) & Fusee gear (2.130+7.140=9.270)
  addPart("gear_gal_io", -121.8, -6.8, 15.2, 0, 0, -fuseeRot * (7.14 / (2.10 * 1.0143)), 1.0143, 1.0143, 1.0, MATS.copper, 1);
  addPart("unitCyl", -121.8, -6.8, 14.8, 0, 0, -fuseeRot * (7.14 / (2.10 * 1.0143)), 0.64, 0.64, 2.0, MATS.bluedSteel, 1);

  // 2) Coaxial Sawtooth Winding Ratchet Wheel (rPitch = 6.50, rTip = 6.95) & Exterior Tangent Click Pawl at [fx, fy - 8.15, 15.85]
  // Law of cosines: pivot R = 8.15, pawl arm L = 4.60 -> angle gamma = acos((8.15^2 + 4.60^2 - 6.50^2)/(2*8.15*4.60)) = 0.9216 rad places beak on rPitch = 6.50!
  addPart("ratchetWheel", fx, fy, 15.85, 0, 0, fuseeRot, 1, 1, 1, MATS.bluedSteel, 1);
  const clickOsc = (Math.PI * 0.5 - 0.9216) + 0.015 * Math.sin(kin.angE * 1.5 * 12);
  addPart("clickPawl", fx, fy - 8.15, 15.85, 0, 0, clickOsc, 1, 1, 1, MATS.brassHi, 1);
  addPart("unitCyl", fx, fy - 8.15, 15.10, 0, 0, 0, 0.72, 0.72, 2.2, MATS.brassLo, 1);
  addPart("unitCyl", fx, fy - 8.15, 16.45, 0, 0, 0, 0.58, 0.58, 0.55, MATS.ruby, 1);
  addPart("slottedScrew", fx - 0.95, fy - 8.95, 16.35, 0, 0, 0.8, 0.48, 0.48, 0.75, MATS.bluedSteel, 1);

  // 3) Pitch-Tangent Great-Wheel-to-Escapement & Shaft-1 Takeoff Spur Train at z = 15.2:
  // Symmetric idler at [-114.5, -21.8, 15.2] (rPitch = 4.20 * 1.2155 = 5.105) tangent to Fusee Great Wheel & under-barrel coaxial transfer wheel (7.14 + 5.105 = 12.245)
  addPart("gear_gal_ca", -114.5, -21.8, 15.2, 0, 0, -fuseeRot * (7.14 / (4.20 * 1.2155)), 1.2155, 1.2155, 1.0, MATS.brassHi, 1);
  addPart("unitCyl", -114.5, -21.8, 14.95, 0, 0, -fuseeRot * (7.14 / (4.20 * 1.2155)), 0.78, 0.78, 1.9, MATS.bluedSteel, 1);
  addPart("oilCup", -114.5, -21.8, 15.88, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 1);
  // Coaxial free-turning transfer wheel under the stationary Mainspring Barrel post (decoupled from barrel drum by a bronze bushing!)
  addPart("gear_wormWheel", mbx, mby, 15.2, 0, 0, fuseeRot, 1.05, 1.05, 0.60, MATS.brassHi, 1);
  // Escapement coupling idler at [-93.98, -12.53, 15.2] (rPitch = 2.80 * 1.1807 = 3.306) tangent to transfer wheel (7.14+3.306=10.446) and Escapement arbor (3.306+2.10=5.406)
  addPart("gear_gal_eu", -93.98, -12.53, 15.2, 0, 0, -fuseeRot * (7.14 / (2.80 * 1.1807)), 1.1807, 1.1807, 1.0, MATS.copper, 1);
  addPart("unitCyl", -93.98, -12.53, 14.95, 0, 0, -fuseeRot * (7.14 / (2.80 * 1.1807)), 0.78, 0.78, 1.9, MATS.bluedSteel, 1);
  addPart("oilCup", -93.98, -12.53, 15.88, 0, 0, 0, 0.65, 0.65, 0.65, MATS.brassHi, 1);

  // 4) Exact Pitch-Tangent Mechanical Bevel Drop-Takeoff at [-80.20, -23.43, z = 9.0..15.9] inside the open recessed driveBox gallery!
  // Idler at [-85.60, -17.50, 15.2] (rPitch = 4.20 * 1.2250 = 5.145) is pitch-tangent to Escapement arbor [-88.80, -11.00] (2.10 + 5.145 = 7.245)
  // and to Takeoff pinion [-80.20, -23.43] (rPitch = 2.10 * 1.3692 = 2.875 -> 5.145 + 2.875 = 8.020), coaxial with Shaft 1 y = -23.43!
  const escArborRot = fuseeRot * (7.14 / 2.10);
  const takeoffArborRot = fuseeRot * (7.14 / (2.10 * 1.3692));
  addPart("gear_gal_ca", -85.60, -17.50, 15.2, 0, 0, -escArborRot * (2.10 / (4.20 * 1.2250)), 1.2250, 1.2250, 0.72, MATS.brass, 1);
  addPart("unitCyl", -85.60, -17.50, 14.8, 0, 0, -escArborRot * (2.10 / (4.20 * 1.2250)), 0.82, 0.82, 1.8, MATS.brassLo, 1);
  addPart("gear_gal_io", -80.20, -23.43, 15.2, 0, 0, takeoffArborRot, 1.3692, 1.3692, 1.0, MATS.copper, 1);
  addPart("unitCyl", -80.20, -23.43, 13.7, 0, 0, takeoffArborRot, 0.78, 0.78, 4.4, MATS.steel, 1);
  addPart("unitCyl", -80.20, -23.43, 9.8, 0, 0, 0, 1.38, 1.38, 2.6, MATS.brassLo, 1);
  addPart("unitCyl", -80.20, -23.43, 10.8, 0, Math.PI * 0.5, 0, 1.28, 1.28, 1.8, MATS.brass, 1);
  addPart("oilCup", -80.20, -23.43, 15.88, 0, 0, 0, 0.65, 0.65, 0.65, MATS.brassHi, 1);
  addPart("bevelGear", -80.20, -23.43, 12.1, Math.PI, 0, -takeoffArborRot, 0.58, 0.58, 0.58, MATS.copper, 1);
  addPart("bevelGear", -78.90, -23.43, 10.8, 0, -Math.PI * 0.5, kin.angE * 4.0, 0.58, 0.58, 0.58, MATS.brassHi, 1);

  addPickSphere(kx, ky, 52.0, 18.0, 1);
  addPickSphere(kx, ky, 26.0, 15.0, 1);
  addPickSphere(fx, fy, 23.0, 12.0, 1);

  // ========================================================================
  // GROUP 14: Da Vinci Crown-Wheel & Verge-Foliot Regulator (Codex Madrid I)
  // ========================================================================
  {
    const [ex, ey] = ESCAPEMENT_XY;

    // 1) Lower Footstep Potence, 3-Screw Leveling Plate, Ruby Cup Jewel & Micrometer Thumbwheel seated on solid driveBox deck
    addPart("unitCyl", ex, ey, 14.25, 0, 0, 0, 2.8, 2.8, 0.55, MATS.brassLo, 14);
    addPart("jewelChaton", ex, ey, 14.65, 0, 0, 0, 0.92, 0.92, 0.55, MATS.brassHi, 14);
    addPart("unitCyl", ex, ey, 14.70, 0, 0, 0, 1.15, 1.15, 0.35, MATS.ruby, 14);
    for (let s = 0; s < 3; s++) {
      const sa = (s / 3) * Math.PI * 2 + 0.3;
      addPart("slottedScrew", ex + Math.cos(sa) * 2.1, ey + Math.sin(sa) * 2.1, 14.55, 0, 0, sa, 0.42, 0.42, 0.55, MATS.bluedSteel, 14);
    }

    // 2) Architectural Overhanging Gallows Suspension Bracket ("Forca di Sospensione", Codex Madrid I f. 27v)
    // Anchored at [ex, ey + 6.5] behind the verge arbor (rz = -pi/2) so the X-aligned foliot beam and 4 poise weights swing in open space!
    addPart("gallowsBracket", ex, ey + 6.5, 14.0, 0, 0, -Math.PI * 0.5, 1, 1, 1, MATS.brass, 14);
    addPart("unitCyl", ex, ey + 6.5, 29.0, 0, 0, 0, 1.05, 1.05, 30.0, MATS.brassLo, 14);
    // Twin Flanking Turned Baluster Columns at [ex +- 3.2, ey + 6.2] behind the foliot oscillation envelope
    for (const sx of [-3.2, 3.2]) {
      addPart("balusterPillar", ex + sx, ey + 6.2, 21.5, 0, 0, 0, 0.52, 0.52, 1.75, MATS.brassHi, 14);
      addPart("unitCyl", ex + sx, ey + 6.2, 14.3, 0, 0, 0, 0.72, 0.72, 0.65, MATS.brassLo, 14);
    }
    addSegmentRod([ex, ey + 6.5, 44.0], [ex, ey - 1.45, 44.0], 1.15, MATS.brassHi, 14);
    // Upper Silk-Cord Suspension Windlass Drum, Ratchet Click & Butterfly Tension Key
    addPart("unitCyl", ex, ey, 44.6, 0, 0, 0, 1.5, 1.5, 0.9, MATS.ruby, 14);
    addPart("oilCup", ex, ey, 45.0, 0, 0, 0, 0.85, 0.85, 0.85, MATS.brassHi, 14);
    addPart("daVinciWingNut", ex, ey - 1.35, 44.0, Math.PI * 0.5, 0, 0.4, 0.65, 0.65, 0.65, MATS.bluedSteel, 14);

    // 3) Vertical Escapement Drive Arbor at [ex - 4.8, ey] coupling Mainspring train (z = 15.2) up to Crown Wheel shaft (z = 26.5)
    addPart("gear_gal_io", ex - 4.8, ey, 15.2, 0, 0, escArborRot, 1.0, 1.0, 1.0, MATS.brassHi, 14);
    addPart("unitCyl", ex - 4.8, ey, 14.2, 0, 0, 0, 1.15, 1.15, 0.4, MATS.brassLo, 14);
    addPart("unitCyl", ex - 4.8, ey, 20.2, 0, 0, escArborRot, 0.64, 0.64, 12.4, MATS.steel, 14);
    addPart("unitCyl", ex - 4.8, ey, 24.1, 0, 0, 0, 1.12, 1.12, 0.5, MATS.brassLo, 14);
    addPart("bevelGear", ex - 4.8, ey, 25.35, 0, 0, kin.crownAngle, 0.48, 0.48, 0.48, MATS.brassHi, 14);
    addPart("bevelGear", ex - 3.6, ey, 26.5, 0, -Math.PI * 0.5, -kin.crownAngle, 0.52, 0.52, 0.52, MATS.copper, 14);

    // 4) Horizontal Crown Wheel ("Ruota a Corona", 20 vertical contrate teeth with true backward recoil during pallet arrest!)
    // Cleanly spaced along the horizontal arbor [ex - 7.4 .. ex - 1.4] on an elevated z = 22.8 sub-bridge anchored to the Gallows column
    addPart("daVinciCrown", ex - 1.6, ey, 26.5, 0, Math.PI * 0.5, kin.crownAngle, 0.92, 0.92, 0.92, MATS.copper, 14);
    addSegmentRod([ex - 7.4, ey, 26.5], [ex - 1.4, ey, 26.5], 0.88, MATS.steel, 14);
    addSegmentRod([ex - 7.4, ey, 22.8], [ex - 2.0, ey, 22.8], 0.46, MATS.brassLo, 14);
    addSegmentRod([ex - 2.2, ey, 22.8], [ex, ey + 6.5, 22.8], 0.46, MATS.brassLo, 14);
    for (const sx of [-7.3, -2.2]) {
      addPart("unitCyl", ex + sx, ey, 24.65, 0, 0, 0, 0.55, 0.55, 3.7, MATS.brassLo, 14);
      addPart("unitCyl", ex + sx, ey, 26.5, 0, Math.PI * 0.5, 0, 1.38, 1.38, 1.05, MATS.brassHi, 14);
      addPart("oilCup", ex + sx, ey, 27.55, 0, 0, 0, 0.55, 0.55, 0.55, MATS.brassHi, 14);
    }
    addPart("unitCyl", ex - 7.45, ey, 26.5, 0, Math.PI * 0.5, 0, 0.82, 0.82, 0.52, MATS.ruby, 14);

    // 5) Vertical Blued-Steel Verge Pallet Arbor, Dovetail Pallet Collets & Physical Upper/Lower Ruby Pallet Blades
    addPart("unitCyl", ex, ey, 29.5, 0, 0, kin.foliotAngle, 0.68, 0.68, 29.5, MATS.bluedSteel, 14);
    // Crown pitch radius is 5.4 * 0.92 = 4.968 -> top & bottom crown teeth sit at z = 26.5 +- 4.97
    addPart("vergePalletCollet", ex, ey, 31.45, 0, 0, kin.foliotAngle + 0.62, 0.82, 0.82, 0.82, MATS.brassHi, 14);
    addPart("vergePalletCollet", ex, ey, 21.55, 0, 0, kin.foliotAngle - 0.62, 0.82, 0.82, 0.82, MATS.brassHi, 14);
    addPart("transferBox", ex - 0.65, ey, 31.45, 0, 0, kin.foliotAngle + 0.62, 0.14, 0.04, 0.18, MATS.ruby, 14);
    addPart("transferBox", ex - 0.65, ey, 21.55, 0, 0, kin.foliotAngle - 0.62, 0.14, 0.04, 0.18, MATS.ruby, 14);
    // Upper & lower verge arbor thrust collars
    addPart("unitCyl", ex, ey, 17.2, 0, 0, kin.foliotAngle, 1.15, 1.15, 0.75, MATS.brassHi, 14);
    addPart("unitCyl", ex, ey, 34.8, 0, 0, kin.foliotAngle, 1.15, 1.15, 0.75, MATS.brassHi, 14);

    // 6) Banking-Pin Bridge ("Coq de Verge"), Sunken Jewel Chaton, Slotted Screw & Graduated Beat-Scale Ring (z = 31.8..34.6, below poise weights!)
    // Oriented at rz = -pi/2 so its foot anchors directly into the rear Gallows column at [ex, ey + 5.7] with zero poise-weight collision!
    addPart("horologicalCock", ex, ey, 31.8, 0, 0, -Math.PI * 0.5, 0.54, 0.68, 0.62, MATS.brass, 14);
    addPart("jewelChaton", ex, ey, 34.45, 0, 0, 0, 0.72, 0.72, 0.75, MATS.brassHi, 14);
    addPart("slottedScrew", ex, ey + 5.7, 32.7, 0, 0, 0.5, 0.55, 0.55, 0.75, MATS.bluedSteel, 14);
    addPart("horizonRing", ex, ey, 34.55, 0, 0, 0, 0.065, 0.065, 0.42, MATS.brassHi, 14);
    addPart("hand_dragon", ex, ey, 34.95, 0, 0, kin.foliotAngle, 0.26, 0.26, 0.55, MATS.bluedSteel, 14);

    // 7) Oscillating Multi-Notch Foliot Balance Beam, Threaded End-Caps & 4 Hanging Poises (with low-profile brass stirrups only at active poises!)
    addPart("daVinciFoliot", ex, ey, 41.2, 0, 0, kin.foliotAngle, 1, 1, 1, MATS.brassHi, 14);
    addPart("daVinciWingNut", ex, ey, 42.65, 0, 0, kin.foliotAngle, 0.78, 0.78, 0.78, MATS.bluedSteel, 14);

    const cosF = Math.cos(kin.foliotAngle);
    const sinF = Math.sin(kin.foliotAngle);
    for (const armSign of [-1, 1]) {
      // Low-profile brass saddle stirrups at the 2 active poise-weight suspension notches per arm (r = +-6.3 and +-11.3)
      for (const rStirrup of [6.3, 11.3]) {
        const nx = ex + armSign * rStirrup * cosF;
        const ny = ey + armSign * rStirrup * sinF;
        addPart("unitCyl", nx, ny, 41.95, 0, 0, kin.foliotAngle, 0.34, 0.34, 0.32, MATS.brassLo, 14);
      }
      // Threaded micro-regulation end-cap & butterfly locknut at arm tip (r = +-13.8), local +Z aligned along [armSign*cosF, armSign*sinF, 0]
      const tipX = ex + armSign * 13.8 * cosF;
      const tipY = ey + armSign * 13.8 * sinF;
      const fArmUx = armSign * cosF, fArmUy = armSign * sinF;
      addPart(
        "daVinciWingNut",
        tipX,
        tipY,
        41.2,
        Math.atan2(-fArmUy, 0),
        Math.asin(Math.max(-1, Math.min(1, fArmUx))),
        0,
        0.58,
        0.58,
        0.58,
        MATS.brassHi,
        14
      );

      // Outer Primary Poise Weight Assembly at r = +-11.3 with suspension rod
      const wxOut = ex + armSign * 11.3 * cosF;
      const wyOut = ey + armSign * 11.3 * sinF;
      addPart("unitCyl", wxOut, wyOut, 39.4, 0, 0, 0, 0.28, 0.28, 5.2, MATS.steel, 14);
      addPart("foliotWeight", wxOut, wyOut, 36.8, 0, 0, kin.foliotAngle, 1.0, 1.0, 1.0, MATS.copper, 14);
      addPart("daVinciWingNut", wxOut, wyOut, 37.6, 0, 0, kin.foliotAngle, 0.52, 0.52, 0.52, MATS.brassHi, 14);

      // Inner Secondary Trimmer Poise Weight Assembly at r = +-6.3 with suspension rod
      const wxIn = ex + armSign * 6.3 * cosF;
      const wyIn = ey + armSign * 6.3 * sinF;
      addPart("unitCyl", wxIn, wyIn, 39.6, 0, 0, 0, 0.26, 0.26, 4.8, MATS.steel, 14);
      addPart("foliotWeight", wxIn, wyIn, 37.6, 0, 0, kin.foliotAngle, 0.68, 0.68, 0.72, MATS.brassHi, 14);
      addPart("daVinciWingNut", wxIn, wyIn, 38.2, 0, 0, kin.foliotAngle, 0.42, 0.42, 0.42, MATS.bluedSteel, 14);
    }

    addPickSphere(ex, ey, 35.0, 14.0, 14);
  }

  // ========================================================================
  // GROUP 2: Center Plate Astrolabe Planisphere, Lower Master Sub-Bridge & Line Shafts
  // ========================================================================
  const [cx, cy] = CENTER_XY;
  const [lx, ly] = LAY_XY;
  const bAng = Math.atan2(ly - cy, lx - cx);
  const bCos = Math.cos(bAng), bSin = Math.sin(bAng);

  addPart("plinthDisc", cx, cy, 0, 0, 0, 0, 1, 1, 1, MATS.gunmetal, 2);
  addPart("plinthRing", cx, cy, 0, 0, 0, 0, 1, 1, 1, MATS.brassLo, 2);
  addPart("astrolabeDisc", cx, cy, 5.05, 0, 0, 0, 1, 1, 1, MATS.astrolabe, 2);
  // Lower Master Coaxial-to-Countershaft Sub-Plate Bridge ("Pont Inférieur de Rouage") flush on the Center Plate (zMax = 5.98 < bevels z = 6.45!)
  addPart("bridgeBar", cx, cy, 5.42, 0, 0, bAng, 1, 1, 0.35, MATS.brassHi, 2);
  // Eccentric Astrolabe Ecliptic Rete Ring (nested in open planisphere field clear of Pillow Block #1 & Countershaft) + 2-Gear Sidereal Drive Train + 4 Flammule Star Pointers
  const reteCx = cx + 4.2, reteCy = cy + 5.2, reteRot = kin.angE * 0.25;
  addPart("jupMoonRing", reteCx, reteCy, 5.42, 0, 0, reteRot, 2.05, 2.05, 0.65, MATS.brassHi, 2);
  addPart("gear_gal_io", reteCx + 12.15, reteCy, 5.48, 0, 0, -reteRot * 4.5, 0.58, 0.58, 0.45, MATS.copper, 2);
  addPart("gear_gal_eu", reteCx + 9.112, reteCy, 5.48, 0, 0, reteRot * 4.5 * (1.218 / 1.82), 0.65, 0.65, 0.45, MATS.brassHi, 2);
  addPart("jewelChaton", reteCx + 12.15, reteCy, 5.75, 0, 0, 0, 0.55, 0.55, 0.48, MATS.brassHi, 2);
  addPart("jewelChaton", reteCx + 9.112, reteCy, 5.75, 0, 0, 0, 0.55, 0.55, 0.48, MATS.brassHi, 2);
  for (let sIdx = 0; sIdx < 4; sIdx++) {
    const flAng = reteRot + (sIdx * Math.PI) / 2 + 0.38;
    const fxR = reteCx + Math.cos(flAng) * 9.85;
    const fyR = reteCy + Math.sin(flAng) * 9.85;
    addPart("hand_eot", fxR, fyR, 5.66, 0, 0, flAng + Math.PI, 0.16, 0.16, 0.38, MATS.brassHi, 2);
    addPart("unitCyl", fxR - Math.cos(flAng) * 1.45, fyR - Math.sin(flAng) * 1.45, 5.72, 0, 0, 0, 0.32, 0.32, 0.28, MATS.ruby, 2);
  }
  const PERIM_SCREW_DEGS = [-4, 8, 44, 54, 86, 96, 130, 140, 172, 232, 268, 304];
  for (const sDeg of PERIM_SCREW_DEGS) {
    const ba = (sDeg * Math.PI) / 180;
    addPart("slottedScrew", cx + Math.cos(ba) * 50.8, cy + Math.sin(ba) * 50.8, 6.25, 0, 0, ba + 0.4, 0.78, 0.78, 0.85, MATS.bluedSteel, 2);
  }
  for (let cIdx = 0; cIdx < 4; cIdx++) {
    const ca = (cIdx / 4) * Math.PI * 2;
    addPart("jewelChaton", cx + Math.cos(ca) * 45.2, cy + Math.sin(ca) * 45.2, 5.35, 0, 0, ca, 0.95, 0.95, 0.75, MATS.brassHi, 2);
    addPart("unitCyl", cx + Math.cos(ca) * 45.2, cy + Math.sin(ca) * 45.2, 5.45, 0, 0, 0, 1.35, 1.35, 0.45, MATS.ruby, 2);
  }

  // Left Inlet Transfer Cage at [-48.5, -23.43] (elevated at z = 6.55 above horizonRing z = 6.45 on a compact waisted pedestal clear of 198° Greek Column!)
  const inBoxX = -48.5, inBoxY = ly;
  const inArborX = inBoxX + 1.2;
  addPart("unitCyl", inArborX, inBoxY, 5.82, 0, 0, 0, 2.85, 2.85, 0.65, MATS.brassLo, 2);
  addPart("lanternCageFrame", inArborX, inBoxY, 6.55, 0, 0, 0, 0.64, 0.58, 0.88, MATS.brass, 2);
  // Da Vinci Lantern Pinion ("Rocchetto a Lanterna") spinning around its +X shaft inside the side window without clipping the architrave
  addPart("daVinciLantern", inArborX - 1.95, inBoxY, 10.8, 0, Math.PI * 0.5, -kin.angE * 4.0, 0.56, 0.56, 0.60, MATS.brassHi, 2);
  addPart("daVinciCrown", inArborX, inBoxY, 8.85, 0, 0, kin.angE * 2.0, 0.56, 0.56, 0.65, MATS.copper, 2);
  // Vertical Crown-Wheel Arbor, Mating 90° Lower Miter Bevel Pair to Shaft 2, Coaxial Top Ruby Thrust Jewel, Chaton, Oil Cup & 4 Corner Butterfly Nuts
  addPart("unitCyl", inArborX, inBoxY, 10.25, 0, 0, kin.angE * 2.0, 0.68, 0.68, 6.8, MATS.steel, 2);
  addPart("bevelGear", inArborX, inBoxY, 6.95, 0, 0, kin.angE * 2.0, 0.56, 0.56, 0.56, MATS.brassHi, 2);
  addPart("bevelGear", inArborX + 1.35, inBoxY, 7.8, 0, -Math.PI * 0.5, -kin.angE * 2.0, 0.56, 0.56, 0.56, MATS.copper, 2);
  addPart("jewelChaton", inArborX, inBoxY, 14.15, 0, 0, 0, 0.82, 0.82, 0.72, MATS.brassHi, 2);
  addPart("unitCyl", inArborX, inBoxY, 14.22, 0, 0, 0, 1.10, 1.10, 0.42, MATS.ruby, 2);
  addPart("oilCup", inArborX, inBoxY, 14.48, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  for (const [nx, ny] of [[-3.0, -3.25], [3.0, -3.25], [-3.0, 3.25], [3.0, 3.25]]) {
    addPart("daVinciWingNut", inArborX + nx, inBoxY + ny, 13.95, 0, 0, 0.3, 0.44, 0.44, 0.44, MATS.brassHi, 2);
  }

  // Shaft 1: Left Line-Shaft at y = ly = -23.43, z = 10.8 from Drive Box [-78.90, -23.43, 10.8] through Pillow Block #0 (x = -63.0, clear of plinthDisc!) to Lantern Pinion
  addPart("unitCyl", -63.0, ly, 4.675, 0, 0, 0, 2.65, 2.65, 7.75, MATS.brassLo, 2);
  addPart("pillowBlock", -63.0, ly, 8.55, 0, 0, 0, 0.72, 0.85, 0.50, MATS.brass, 2);
  addSegmentRod([-78.90, ly, 10.8], [inArborX - 1.95, inBoxY, 10.8], 0.92, MATS.steel, 2);
  // Coaxial Bolted Split-Flange Coupling, Drive-Box Exit Gland, Twin Thrust Collars & Gravity Oil Cup on Shaft 1
  addPart("unitCyl", -74.8, ly, 10.8, 0, Math.PI * 0.5, 0, 1.95, 1.95, 1.1, MATS.brassLo, 2);
  addPart("flangedCoupling", -70.6, ly, 10.8, 0, Math.PI * 0.5, -kin.angE * 4.0, 0.82, 0.82, 0.82, MATS.brassHi, 2);
  addPart("unitCyl", -65.2, ly, 10.8, 0, Math.PI * 0.5, -kin.angE * 4.0, 1.58, 1.58, 0.60, MATS.bluedSteel, 2);
  addPart("unitCyl", -60.8, ly, 10.8, 0, Math.PI * 0.5, -kin.angE * 4.0, 1.58, 1.58, 0.60, MATS.bluedSteel, 2);
  addPart("oilCup", -63.0, ly, 12.35, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  addPart("daVinciWingNut", -63.0, ly - 1.48, 11.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPart("daVinciWingNut", -63.0, ly + 1.48, 11.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);

  // Shaft 2: Transfer shaft at y = ly = -23.43, z = 7.8 from Inlet Cage through Pillow Block #1 (x = -35.6, tz = 5.55, sz = 0.50 -> bore z = 7.80 exact!)
  // Base tz = 5.55 sits strictly on top of astrolabeDisc (z = 5.50) with zero sunken collision!
  addPart("pillowBlock", -35.6, inBoxY, 5.55, 0, 0, 0, 0.72, 0.85, 0.50, MATS.brass, 2);
  addSegmentRod([inArborX + 1.35, inBoxY, 7.8], [lx - 2.45, ly, 7.8], 1.18, MATS.steel, 2);
  addPart("bevelGear", lx - 2.45, ly, 7.8, 0, Math.PI * 0.5, kin.angE * 2.0, 0.62, 0.62, 0.62, MATS.copper, 2);
  addPart("bevelGear", lx, ly, 6.45, 0, 0, -kin.angE, 0.74, 0.74, 0.62, MATS.brassHi, 2);
  // Coaxial Split-Flange Coupling (x = -40.2), Twin Thrust Collars (x = -37.8 & -33.4) & Oil Cup on Shaft 2
  addPart("flangedCoupling", -40.2, inBoxY, 7.8, 0, Math.PI * 0.5, kin.angE * 2.0, 0.76, 0.76, 0.76, MATS.brassHi, 2);
  addPart("unitCyl", -37.8, inBoxY, 7.8, 0, Math.PI * 0.5, kin.angE * 2.0, 1.50, 1.50, 0.55, MATS.bluedSteel, 2);
  addPart("unitCyl", -33.4, inBoxY, 7.8, 0, Math.PI * 0.5, kin.angE * 2.0, 1.50, 1.50, 0.55, MATS.bluedSteel, 2);
  addPart("oilCup", -35.6, inBoxY, 9.35, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  addPart("daVinciWingNut", -35.6, inBoxY - 1.48, 8.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPart("daVinciWingNut", -35.6, inBoxY + 1.48, 8.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPickSphere(-63.0, ly, 10.8, 12, 2);
  addPickSphere(inBoxX, inBoxY, 10.0, 11, 2);

  // Shaft 3: Front Rack Takeoff Shaft along x = lx = -13.2, elevated to z = 7.80 (strictly above astrolabeDisc z=5.50, jupMoonRing z=5.63,
  // plinthRing z=6.20, and horizonRing z=6.45, and strictly below gear_lay_M zMin=10.65!) stopping at y = -64.8 behind Vernier carriage (y <= -69.94)
  addPart("unitCyl", lx, -59.0, 2.775, 0, 0, 0, 2.65, 2.65, 5.55, MATS.brassLo, 2);
  addPart("pillowBlock", lx, -59.0, 5.55, 0, 0, Math.PI * 0.5, 0.72, 0.85, 0.50, MATS.brass, 2);
  addPart("daVinciWingNut", lx - 1.48, -59.0, 8.78, 0, 0, Math.PI * 0.5, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPart("daVinciWingNut", lx + 1.48, -59.0, 8.78, 0, 0, Math.PI * 0.5, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addSegmentRod([lx, ly - 2.45, 7.8], [lx, -64.8, 7.8], 1.05, MATS.steel, 2);
  // Front Takeoff Shaft 3 bevel gear at [lx, ly - 2.45, 7.8] meshing at 90° with Countershaft base bevel gear [lx, ly, 6.45]
  addPart("bevelGear", lx, ly - 2.45, 7.8, -Math.PI * 0.5, 0, kin.angE * 2.0, 0.62, 0.62, 0.62, MATS.copper, 2);
  // Vertical Miter-Bevel Drop Arbor at [lx, -66.2, z = 3.25..7.8] coupling Shaft 3 down to rear rack transfer shaft [lx + 1.4, -66.2, 3.25] -> [53.5, -66.2, 3.25]
  addPart("unitCyl", lx, -66.2, 2.15, 0, 0, 0, 1.95, 1.95, 2.7, MATS.brassLo, 2);
  addPart("unitCyl", lx, -66.2, 5.15, 0, 0, kin.angE * 4.0, 0.72, 0.72, 4.2, MATS.steel, 2);
  addPart("bevelGear", lx, -64.8, 7.8, Math.PI * 0.5, 0, -kin.angE * 2.0, 0.58, 0.58, 0.58, MATS.brassHi, 2);
  addPart("bevelGear", lx, -66.2, 6.6, 0, 0, kin.angE * 4.0, 0.58, 0.58, 0.58, MATS.copper, 2);
  addPart("bevelGear", lx, -66.2, 4.45, Math.PI, 0, -kin.angE * 4.0, 0.58, 0.58, 0.58, MATS.brassHi, 2);
  addPart("bevelGear", lx + 1.4, -66.2, 3.25, 0, -Math.PI * 0.5, kin.angE * 16.0, 0.58, 0.58, 0.58, MATS.copper, 2);
  addSegmentRod([lx + 1.4, -66.2, 3.25], [53.5, -66.2, 3.25], 0.72, MATS.steel, 13);
  addPart("unitCyl", 53.5, -66.2, 2.2, 0, 0, 0, 0.72, 0.72, 2.8, MATS.brassLo, 13);
  addPart("gear_gal_ca", 53.5, -66.2, 3.25, 0, Math.PI * 0.5, -kin.angE * 16.0, 1.119, 1.119, 0.85, MATS.brassHi, 13);
  addPart("gear_gal_ca", 53.5, rackY - 1.6, 3.25, 0, Math.PI * 0.5, kin.angE * 16.0, 1.119, 1.119, 0.85, MATS.copper, 13);
  // Coaxial Split-Flange Coupling (z = 7.8, clearing astrolabeDisc & jupMoonRing!), Thrust Collars & Oil Cup on Front Takeoff Shaft 3
  addPart("flangedCoupling", lx, -38.5, 7.8, Math.PI * 0.5, 0, -kin.angE * 2.0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  addPart("unitCyl", lx, -56.4, 7.8, Math.PI * 0.5, 0, -kin.angE * 2.0, 1.45, 1.45, 0.55, MATS.bluedSteel, 2);
  addPart("unitCyl", lx, -61.6, 7.8, Math.PI * 0.5, 0, -kin.angE * 2.0, 1.45, 1.45, 0.55, MATS.bluedSteel, 2);
  addPart("oilCup", lx, -59.0, 9.35, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);

  // Shaft 4 & Shaft 5: Right Outlet Transfer Cage at [30.2, -23.43] (elevated at z = 6.55 above horizonRing z = 6.45 on a compact waisted pedestal!)
  const outBoxX = 30.2, outBoxY = ly;
  const outArborX = outBoxX - 1.2;
  addPart("unitCyl", outArborX, outBoxY, 5.82, 0, 0, 0, 2.85, 2.85, 0.65, MATS.brassLo, 2);
  addPart("lanternCageFrame", outArborX, outBoxY, 6.55, 0, 0, 0, 0.64, 0.58, 0.88, MATS.brass, 2);
  addPart("bevelGear", lx + 2.45, ly, 7.8, 0, -Math.PI * 0.5, kin.angE * 2.0, 0.62, 0.62, 0.62, MATS.copper, 2);
  addPart("bevelGear", outArborX - 1.35, outBoxY, 7.8, 0, Math.PI * 0.5, -kin.angE * 2.0, 0.56, 0.56, 0.56, MATS.brassHi, 2);
  addSegmentRod([lx + 2.45, ly, 7.8], [outArborX - 1.35, outBoxY, 7.8], 1.18, MATS.steel, 2);
  // Coaxial Split-Flange Coupling & Twin Thrust Collars on Shaft 4 (all at x >= 11.2 strictly outside gear_lay_M x = +6.14!)
  addPart("flangedCoupling", 16.5, outBoxY, 7.8, 0, Math.PI * 0.5, -kin.angE * 2.0, 0.80, 0.80, 0.80, MATS.brassHi, 2);
  addPart("unitCyl", 11.2, outBoxY, 7.8, 0, Math.PI * 0.5, -kin.angE * 2.0, 1.52, 1.52, 0.65, MATS.bluedSteel, 2);
  addPart("unitCyl", 21.8, outBoxY, 7.8, 0, Math.PI * 0.5, -kin.angE * 2.0, 1.52, 1.52, 0.65, MATS.bluedSteel, 2);

  addPart("daVinciCrown", outArborX, outBoxY, 8.85, 0, 0, kin.angE * 2.0, 0.56, 0.56, 0.65, MATS.brassHi, 2);
  addPart("daVinciLantern", outArborX + 1.95, outBoxY, 10.8, 0, -Math.PI * 0.5, -kin.angE * 4.0, 0.56, 0.56, 0.60, MATS.copper, 2);
  // Vertical Crown Arbor, Lower Miter Bevel, Coaxial Top Ruby Thrust Jewel, Chaton, Oil Cup & 4 Corner Butterfly Nuts inside outBox
  addPart("unitCyl", outArborX, outBoxY, 10.25, 0, 0, kin.angE * 2.0, 0.68, 0.68, 6.8, MATS.steel, 2);
  addPart("bevelGear", outArborX, outBoxY, 6.95, 0, 0, kin.angE * 2.0, 0.56, 0.56, 0.56, MATS.copper, 2);
  addPart("jewelChaton", outArborX, outBoxY, 14.15, 0, 0, 0, 0.82, 0.82, 0.72, MATS.brassHi, 2);
  addPart("unitCyl", outArborX, outBoxY, 14.22, 0, 0, 0, 1.10, 1.10, 0.42, MATS.ruby, 2);
  addPart("oilCup", outArborX, outBoxY, 14.48, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  for (const [nx, ny] of [[-3.0, -3.25], [3.0, -3.25], [-3.0, 3.25], [3.0, 3.25]]) {
    addPart("daVinciWingNut", outArborX + nx, outBoxY + ny, 13.95, 0, 0, -0.3, 0.44, 0.44, 0.44, MATS.brassHi, 2);
  }

  addPart("unitCyl", 49.5, outBoxY, 4.675, 0, 0, 0, 2.65, 2.65, 7.75, MATS.brassLo, 2);
  addPart("pillowBlock", 49.5, outBoxY, 8.55, 0, 0, 0, 0.72, 0.85, 0.50, MATS.brass, 2);
  addSegmentRod([outArborX + 1.95, outBoxY, 10.8], [64.0, outBoxY, 10.8], 0.92, MATS.steel, 2);
  // Coaxial Split-Flange Coupling, Ephemeris-Box Entry Gland, Twin Thrust Collars & Oil Cup on Shaft 5
  addPart("flangedCoupling", 40.8, outBoxY, 10.8, 0, Math.PI * 0.5, kin.angE * 4.0, 0.82, 0.82, 0.82, MATS.brassHi, 2);
  addPart("unitCyl", 47.3, outBoxY, 10.8, 0, Math.PI * 0.5, kin.angE * 4.0, 1.58, 1.58, 0.60, MATS.bluedSteel, 2);
  addPart("unitCyl", 51.7, outBoxY, 10.8, 0, Math.PI * 0.5, kin.angE * 4.0, 1.58, 1.58, 0.60, MATS.bluedSteel, 2);
  addPart("oilCup", 49.5, outBoxY, 12.35, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 2);
  addPart("daVinciWingNut", 49.5, outBoxY - 1.48, 11.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPart("daVinciWingNut", 49.5, outBoxY + 1.48, 11.78, 0, 0, 0, 0.44, 0.44, 0.44, MATS.bluedSteel, 2);
  addPart("unitCyl", 59.8, outBoxY, 10.8, 0, Math.PI * 0.5, 0, 1.95, 1.95, 1.1, MATS.brassLo, 2);
  addPickSphere(49, outBoxY, 10.8, 12, 2);

  // ========================================================================
  // GROUP 3 & GROUP 4: 6-Level True Coaxial Telescoping Keplerian Spur-Gear Train
  // ========================================================================
  const exp = (explodeMode || 0) * 0.55;
  // Flush countershaft lower bearing bezel on the Center Plate (z = 5.05..5.65, leaving the 4-way bevel junction at z = 6.45..8.8 100% unburied!)
  addPart("unitCyl", lx, ly, 5.35, 0, 0, 0, 3.2, 3.2, 0.60, MATS.brassLo, 3);
  addPart("unitCyl", cx, cy, 6.4, 0, 0, 0, 8.2, 8.2, 2.8, MATS.brassLo, 4);

  // Stepped Swiss Countershaft Arbor:
  // 1) Lower slim bevel journal (r = 1.12, z = 5.2..10.4) passing cleanly inside the 4-way bevel differential at z = 6.45..7.80
  // 2) Upper splined gear seat (r = 2.55, z = 10.4..37.2) carrying the 6 Keplerian countershaft gears
  addPart("unitCyl", lx, ly, 7.8, 0, 0, -kin.angE, 1.12, 1.12, 5.2, MATS.steel, 3);
  addPart("unitCyl", lx, ly, 23.8 + exp * 7.0, 0, 0, -kin.angE, 2.55, 2.55, 26.8 + exp * 14.0, MATS.steel, 3);

  // Ordered Bottom-to-Top (elevated to z = 12.0..32.0 so Level 1 gear_lay_M at z = 12.0 clears Shaft 2, Shaft 4 & Pillow Block #1 by > 1.8mm!):
  // Level 1 (Bottom, z = 12.0): Mercury (62T : 16T) -> Innermost sleeve r = 1.65
  // Level 2 (z = 16.0): Venus (48T : 30T) -> 2nd sleeve r = 2.25
  // Level 3 (z = 20.0): Earth (40T : 40T) -> 3rd sleeve r = 3.00
  // Level 4 (z = 24.0): Mars (28T : 52T) -> 4th sleeve r = 3.95
  // Level 5 (z = 28.0): Jupiter (14T : 58T) -> 5th sleeve r = 5.15
  // Level 6 (Top, z = 32.0): Saturn (10T : 64T) -> Outermost 6th sleeve r = 6.70
  const LEVELS = [
    { z: 12.0 + exp * 0.0,  meshL: "gear_lay_M",    meshC: "gear_cen_M",    angC: kin.angM,    teethL: 62, teethC: 16, matL: MATS.brassHi, matC: MATS.copper,  cutC: 0 },
    { z: 16.0 + exp * 2.5,  meshL: "gear_lay_V",    meshC: "gear_cen_V",    angC: kin.angV,    teethL: 48, teethC: 30, matL: MATS.copper,  matC: MATS.brassHi, cutC: 0 },
    { z: 20.0 + exp * 5.0,  meshL: "gear_lay_E",    meshC: "gear_cen_E",    angC: kin.angE,    teethL: 40, teethC: 40, matL: MATS.brassHi, matC: MATS.brass,   cutC: 1 },
    { z: 24.0 + exp * 7.5,  meshL: "gear_lay_Mars", meshC: "gear_cen_Mars", angC: kin.angMars, teethL: 28, teethC: 52, matL: MATS.copper,  matC: MATS.brassHi, cutC: 1 },
    { z: 28.0 + exp * 10.0, meshL: "gear_lay_Jup",  meshC: "gear_cen_Jup",  angC: kin.angJup,  teethL: 14, teethC: 58, matL: MATS.brassHi, matC: MATS.copper,  cutC: 1 },
    { z: 32.0 + exp * 12.5, meshL: "gear_lay_Sat",  meshC: "gear_cen_Sat",  angC: kin.angSat,  teethL: 10, teethC: 64, matL: MATS.copper,  matC: MATS.brassHi, cutC: 1 },
  ];

  const meshAz = Math.atan2(cy - ly, cx - lx);
  for (let i = 0; i < LEVELS.length; i++) {
    const lv = LEVELS[i];
    const rotC = lv.angC;
    const rotL = meshAz - (rotC - meshAz + Math.PI) * (lv.teethC / lv.teethL) + Math.PI / lv.teethL;
    addPart(lv.meshL, lx, ly, lv.z, 0, 0, rotL, 1, 1, 1, lv.matL, 3);
    addPart(lv.meshC, cx, cy, lv.z, 0, 0, rotC, 1, 1, 1, lv.matC, 4, lv.cutC);
    // Spring-loaded anti-backlash split-gear tensioning ring & spiral preload spring seated flush on primary Countershaft gear webs
    if (i < 4) {
      const rSplitPitch = [19.08, 14.77, 12.00, 8.40][i] * 0.62;
      addPart("hairspring", lx, ly, lv.z + 0.62, 0, 0, rotL + 0.4, 0.72 + (3 - i) * 0.14, 0.72 + (3 - i) * 0.14, 0.85, MATS.bluedSteel, 3);
      addPart("slottedScrew", lx + Math.cos(rotL) * rSplitPitch, ly + Math.sin(rotL) * rSplitPitch, lv.z + 0.72, 0, 0, rotL, 0.52, 0.52, 0.75, MATS.bluedSteel, 3);
    }
  }
  addPickSphere(lx, ly, 22.0, 16.5, 3);
  addPickSphere(cx, cy, 22.0, 15.0, 4);

  // Sculpted Chamfered Swiss Train Bridge ("Pont de Rouage Suisse") spanning [cx, cy] and [lx, ly] at z = 35.4,
  // anchored to the mainplate by two outboard turned baluster pillar posts (local x = 12.6, y = +-22.2) strictly outside all 12 gears!
  const zBridge = 35.4 + exp * 14.0;
  addPart("swissTrainBridge", cx, cy, zBridge, 0, 0, bAng, 1, 1, 1, MATS.brassHi, 3, 1);
  const pilH = (zBridge - 1.6) - 5.2;
  for (const sy of [-1, 1]) {
    const earWorldX = cx + 12.6 * bCos - (sy * 22.2) * bSin;
    const earWorldY = cy + 12.6 * bSin + (sy * 22.2) * bCos;
    addPart("balusterPillar", earWorldX, earWorldY, 5.2 + pilH * 0.5, 0, 0, 0, 0.72, 0.72, pilH / 8.0, MATS.brass, 3);
    addPart("unitCyl", earWorldX, earWorldY, 21.2 + exp * 7.0, 0, 0, 0, 0.58, 0.58, 32.0 + exp * 14.0, MATS.steel, 3);
    addPart("jewelChaton", earWorldX, earWorldY, zBridge + 1.65, 0, 0, bAng, 0.92, 0.92, 0.75, MATS.brassHi, 3);
    addPart("slottedScrew", earWorldX, earWorldY, zBridge + 1.82, 0, 0, bAng + sy * 0.6, 0.95, 0.95, 0.95, MATS.bluedSteel, 3);
    // Locating steady-pin ("Pied de Positionnement") projecting from top of bridge beside each outboard screw
    const pinWorldX = cx + 12.6 * bCos - (sy * 19.4) * bSin;
    const pinWorldY = cy + 12.6 * bSin + (sy * 19.4) * bCos;
    addPart("unitCyl", pinWorldX, pinWorldY, zBridge + 2.1, 0, 0, 0, 0.34, 0.34, 0.45, MATS.chrome, 3);
  }
  // Sunken brass-set ruby jewel chaton, thrust jewel & 3 chaton screws on Countershaft bearing [lx, ly] + Annular Central Sleeve Bezel [cx, cy]
  addPart("jewelChaton", lx, ly, zBridge + 1.95, 0, 0, bAng, 1.85, 1.85, 1.2, MATS.brassHi, 3);
  addPart("unitCyl", lx, ly, zBridge + 2.25, 0, 0, 0, 1.45, 1.45, 0.55, MATS.ruby, 3);
  addPart("oilCup", lx, ly, zBridge + 2.50, 0, 0, 0, 0.75, 0.75, 0.75, MATS.brassHi, 3);
  for (let k = 0; k < 3; k++) {
    const ca = bAng + (k * Math.PI * 2) / 3;
    addPart("slottedScrew", lx + Math.cos(ca) * 3.3, ly + Math.sin(ca) * 3.3, zBridge + 1.75, 0, 0, ca, 0.55, 0.55, 0.75, MATS.bluedSteel, 3);
  }
  // Annular copper coaxial bearing bezel (rOut = 8.6, rIn = 8.6 * 0.78 = 6.708 > Saturn sleeve 6.70!)
  addPart("unitTube", cx, cy, zBridge + 1.95, 0, 0, 0, 8.65, 8.65, 0.75, MATS.copper, 4, 1);

  // ========================================================================
  // GROUP 4: 6 True Telescoping Coaxial Tubular Sleeves with Slotted Split-Collets & Roller-Thrust Cages
  // ========================================================================
  const zCollarSat  = 65.0 + exp * 16.0;
  const zCollarJup  = 69.0 + exp * 20.0;
  const zCollarMars = 73.0 + exp * 24.0;
  const zCollarE    = 77.5 + exp * 28.0;
  const zCollarV    = 82.0 + exp * 32.0;
  const zCollarM    = 86.5 + exp * 36.0;

  const SLEEVE_SPECS = [
    { zBot: LEVELS[5].z, zTop: zCollarSat,  rSleeve: 6.70, rCollar: 7.8, hCollar: 2.6, ang: kin.angSat,  matS: MATS.brass,  matC: MATS.brassHi, cut: 1 },
    { zBot: LEVELS[4].z, zTop: zCollarJup,  rSleeve: 5.15, rCollar: 6.3, hCollar: 2.6, ang: kin.angJup,  matS: MATS.copper, matC: MATS.brassHi, cut: 1 },
    { zBot: LEVELS[3].z, zTop: zCollarMars, rSleeve: 3.95, rCollar: 5.0, hCollar: 2.6, ang: kin.angMars, matS: MATS.brass,  matC: MATS.brassHi, cut: 1 },
    { zBot: LEVELS[2].z, zTop: zCollarE,    rSleeve: 3.00, rCollar: 4.1, hCollar: 2.6, ang: kin.angE,    matS: MATS.steel,  matC: MATS.brassHi, cut: 1 },
    { zBot: LEVELS[1].z, zTop: zCollarV,    rSleeve: 2.25, rCollar: 3.2, hCollar: 2.5, ang: kin.angV,    matS: MATS.brass,  matC: MATS.copper,  cut: 1 },
    { zBot: LEVELS[0].z, zTop: zCollarM,    rSleeve: 1.65, rCollar: 2.4, hCollar: 2.4, ang: kin.angM,    matS: MATS.steel,  matC: MATS.brassHi, cut: 0 },
  ];

  for (const sl of SLEEVE_SPECS) {
    const sH = sl.zTop - sl.zBot;
    addPart("unitTube", cx, cy, sl.zBot + sH * 0.5, 0, 0, sl.ang, sl.rSleeve, sl.rSleeve, sH, sl.matS, 4, sl.cut);
    addPart("splitCollet", cx, cy, sl.zTop, 0, 0, sl.ang, sl.rCollar, sl.rCollar, sl.hCollar, sl.matC, 4, sl.cut);
    // Tangential blued-steel pinch screw tightening the split-collet clamping ears (axis [-sin(ang), cos(ang), 0] orthogonal to gap!)
    const pxCol = cx + Math.cos(sl.ang) * (sl.rCollar * 1.04);
    const pyCol = cy + Math.sin(sl.ang) * (sl.rCollar * 1.04);
    const tUx = -Math.sin(sl.ang), tUy = Math.cos(sl.ang);
    addPart(
      "slottedScrew",
      pxCol,
      pyCol,
      sl.zTop,
      Math.atan2(-tUy, 0),
      Math.asin(Math.max(-1, Math.min(1, tUx))),
      0,
      0.42,
      0.42,
      0.95,
      MATS.bluedSteel,
      4,
      sl.cut
    );
    // Miniature bronze roller-thrust bearing cage seated at the top of each coaxial sleeve step
    addPart("rollerThrustCage", cx, cy, sl.zTop + sl.hCollar * 0.56, 0, 0, sl.ang * 0.5, sl.rSleeve * 1.18, sl.rSleeve * 1.18, 1.0, MATS.copper, 4, sl.cut);
  }

  // Stationary Sun Reference Gear below Earth Arm (z = zCollarE - 2.6, rPitch = 5.625, rBore = 3.15 > 3.00) for Tellurian 5-Gear Train
  addPart("gear_tellurian_cen", cx, cy, zCollarE - 2.6, 0, 0, 0, 1, 1, 1, MATS.copper, 8);

  // Central Spindle 0: Sun Spindle (r = 1.10, z = 6.0 .. 103.0 + exp*42)
  const sunSpin = (kin.day / T_SUN_ROT) * Math.PI * 2;
  const sunSpindleH = 97.0 + exp * 40.0;
  addPart("unitCyl", cx, cy, 6.0 + sunSpindleH * 0.5, 0, 0, sunSpin, 1.10, 1.10, sunSpindleH, MATS.chrome, 4);
  addPickSphere(cx, cy, 56.0, 12.0, 4);

  // ========================================================================
  // GROUP 5: Central Rotating Sun & Volumetric Corona
  // ========================================================================
  const sunZ = ECLIPTIC_PLANE_Z + exp * 40.0;
  addPart("balusterPillar", cx, cy, sunZ - 9.8, 0, 0, 0, 1.35, 1.35, 0.75, MATS.brassHi, 5);
  addPart("unitSphere", SUN_POS[0], SUN_POS[1], sunZ, 0, 0, sunSpin, SUN_R, SUN_R, SUN_R, MATS.sun, 5);
  addPart("unitCyl", cx, cy, sunZ + SUN_R + 0.9, 0, 0, 0, 1.5, 1.5, 1.8, MATS.brassHi, 5);
  addPickSphere(SUN_POS[0], SUN_POS[1], sunZ, SUN_R + 3.0, 5);

  // Volumetric Solar Corona Camera-Facing Billboard
  {
    const vx = eyePos[0] - SUN_POS[0], vy = eyePos[1] - SUN_POS[1], vz = eyePos[2] - sunZ;
    const vlen = Math.hypot(vx, vy, vz) || 1;
    const nx = vx / vlen, ny = vy / vlen, nz = vz / vlen;
    let ux = -ny, uy = nx, uz = 0;
    const ulen = Math.hypot(ux, uy) || 1;
    ux /= ulen; uy /= ulen;
    const wx = ny * uz - nz * uy, wy = nz * ux - nx * uz, wz = nx * uy - ny * ux;
    const cRad = SUN_R * 3.55;
    const bx = SUN_POS[0] - nx * 1.2, by = SUN_POS[1] - ny * 1.2, bz = sunZ - nz * 1.2;
    const p00 = [bx - ux * cRad - wx * cRad, by - uy * cRad - wy * cRad, bz - uz * cRad - wz * cRad];
    const p10 = [bx + ux * cRad - wx * cRad, by + uy * cRad - wy * cRad, bz + uz * cRad - wz * cRad];
    const p11 = [bx + ux * cRad + wx * cRad, by + uy * cRad + wy * cRad, bz + uz * cRad + wz * cRad];
    const p01 = [bx - ux * cRad + wx * cRad, by - uy * cRad + wy * cRad, bz - uz * cRad + wz * cRad];
    const cAlpha = lightTheme ? 0.52 : 0.78;
    pushOverlayVert(p00[0], p00[1], p00[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 0.0, 0.8);
    pushOverlayVert(p10[0], p10[1], p10[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 1.0, 0.8);
    pushOverlayVert(p11[0], p11[1], p11[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 1.0, 1.2);

    pushOverlayVert(p00[0], p00[1], p00[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 0.0, 0.8);
    pushOverlayVert(p11[0], p11[1], p11[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 1.0, 1.2);
    pushOverlayVert(p01[0], p01[1], p01[2], nx, ny, nz, 1.0, 0.82, 0.36, cAlpha, 0.0, 1.2);
  }

  // ========================================================================
  // GROUP 6: Mercury Stepped Carrier Arm (z = 68.0, R = 17.5, i = 7.00°)
  // ========================================================================
  {
    const R_M = 17.5, zArm = zCollarM;
    const incM = (7.0 * Math.PI) / 180;
    const zOrb = ECLIPTIC_PLANE_Z + exp * 36.0 + Math.sin(kin.angM) * R_M * Math.sin(incM);
    const mx = cx + Math.cos(kin.angM) * R_M;
    const my = cy + Math.sin(kin.angM) * R_M;
    const bx = cx - Math.cos(kin.angM) * 9.0;
    const by = cy - Math.sin(kin.angM) * 9.0;

    addPart("carrier_M", cx, cy, zArm, 0, 0, kin.angM, 1, 1, 1, MATS.brass, 6);
    addPart("unitCyl", bx, by, zArm + 2.5, 0, 0, 0, 0.85, 0.85, 4.0, MATS.steel, 6);
    addPart("foliotWeight", bx, by, zArm + 3.2, 0, 0, kin.angM, 0.65, 0.65, 0.65, MATS.copper, 6);
    addPart("daVinciWingNut", bx, by, zArm + 4.1, 0, 0, kin.angM, 0.42, 0.42, 0.42, MATS.bluedSteel, 6);
    addPart("unitSphere", bx, by, zArm + 5.0, 0, 0, 0, 1.65, 1.65, 1.65, MATS.brassHi, 6);
    addSegmentRod(
      [cx + Math.cos(kin.angM) * 2.4, cy + Math.sin(kin.angM) * 2.4, zArm + 1.15],
      [cx + Math.cos(kin.angM) * 14.5, cy + Math.sin(kin.angM) * 14.5, zArm + 0.65],
      0.18,
      MATS.bluedSteel,
      6
    );

    const rH = zOrb - 2.55 - (zArm + 1.2);
    addPart("unitCyl", mx, my, zArm + 1.2 + rH * 0.5, 0, 0, kin.angM * 1.5, 1.0, 1.0, rH, MATS.steel, 6);
    addPart("balusterPillar", mx, my, zArm + 1.2, 0, 0, 0, 0.78, 0.78, 0.72, MATS.brassHi, 6);
    // Antikythera Fragment D Mercury High-Eccentricity (e = 0.2056) Apsidal Epicyclic Gear Pair & Ruby Follower Roller
    const mPinX = mx - Math.cos(kin.angM) * 4.06;
    const mPinY = my - Math.sin(kin.angM) * 4.06;
    addSegmentRod([mx, my, zArm + 1.9], [mPinX, mPinY, zArm + 1.9], 0.28, MATS.brassLo, 6);
    addPart("gear_jup_ga", mx, my, zArm + 2.4, 0, 0, kin.angM * 1.5, 0.80, 0.80, 0.52, MATS.brassHi, 6);
    addPart("gear_gal_io", mPinX, mPinY, zArm + 2.4, 0, 0, -kin.angM * 1.5 * (2.80 / 1.26), 0.60, 0.60, 0.52, MATS.copper, 6);
    addPart("unitCyl", mPinX, mPinY, zArm + 2.4, 0, 0, -kin.angM * 1.5 * (2.80 / 1.26), 0.34, 0.34, 0.85, MATS.ruby, 6);
    addPart("unitSphere", mx, my, zOrb, 0, 0, kin.angM * 1.5, 2.55, 2.55, 2.55, MATS.mercury, 6);
    addPickSphere(mx, my, zOrb, 5.5, 6);
    addOrbitRibbon(cx, cy, ECLIPTIC_PLANE_Z + exp * 36.0, R_M, 0.36, incM, 0.0, [0.78, 0.75, 0.68], lightTheme ? 0.34 : 0.28, 64);
    addVelocityTailArc(cx, cy, ECLIPTIC_PLANE_Z + exp * 36.0, R_M, 0.95, kin.angM, 0.85, incM, 0.0, [0.95, 0.88, 0.72], lightTheme ? 0.52 : 0.68, 24);
  }

  // ========================================================================
  // GROUP 7: Venus Stepped Carrier Arm (z = 61.5, R = 29.5, i = 3.39°)
  // ========================================================================
  {
    const R_V = 29.5, zArm = zCollarV;
    const incV = (3.39 * Math.PI) / 180;
    const zOrb = ECLIPTIC_PLANE_Z + exp * 32.0 + Math.sin(kin.angV - 0.6) * R_V * Math.sin(incV);
    const vx = cx + Math.cos(kin.angV) * R_V;
    const vy = cy + Math.sin(kin.angV) * R_V;
    const bx = cx - Math.cos(kin.angV) * 11.5;
    const by = cy - Math.sin(kin.angV) * 11.5;

    addPart("carrier_V", cx, cy, zArm, 0, 0, kin.angV, 1, 1, 1, MATS.copper, 7);
    addPart("unitCyl", bx, by, zArm + 2.8, 0, 0, 0, 0.95, 0.95, 4.5, MATS.steel, 7);
    addPart("foliotWeight", bx, by, zArm + 3.6, 0, 0, kin.angV, 0.74, 0.74, 0.72, MATS.brassHi, 7);
    addPart("daVinciWingNut", bx, by, zArm + 4.6, 0, 0, kin.angV, 0.46, 0.46, 0.46, MATS.bluedSteel, 7);
    addPart("unitSphere", bx, by, zArm + 5.6, 0, 0, 0, 1.95, 1.95, 1.95, MATS.copper, 7);
    addSegmentRod(
      [cx + Math.cos(kin.angV) * 3.2, cy + Math.sin(kin.angV) * 3.2, zArm + 1.2],
      [cx + Math.cos(kin.angV) * 25.0, cy + Math.sin(kin.angV) * 25.0, zArm + 0.7],
      0.20,
      MATS.bluedSteel,
      7
    );

    const rH = zOrb - 3.5 - (zArm + 1.3);
    addPart("unitCyl", vx, vy, zArm + 1.3 + rH * 0.5, 0, 0, 5.0 * (kin.angV - kin.angE), 1.15, 1.15, rH, MATS.steel, 7);
    addPart("balusterPillar", vx, vy, zArm + 1.3, 0, 0, 0, 0.88, 0.88, 0.85, MATS.brassHi, 7);
    // Antikythera Fragment D Venus 5:8 Synodic Pentagram Epicyclic Gear Pair & Ruby Indexing Roller
    const vPinX = vx - Math.cos(kin.angV) * 4.83;
    const vPinY = vy - Math.sin(kin.angV) * 4.83;
    addSegmentRod([vx, vy, zArm + 2.05], [vPinX, vPinY, zArm + 2.05], 0.30, MATS.brassLo, 7);
    addPart("gear_jup_ca", vx, vy, zArm + 2.6, 0, 0, 5.0 * (kin.angV - kin.angE), 0.75, 0.75, 0.55, MATS.brassHi, 7);
    addPart("gear_gal_eu", vPinX, vPinY, zArm + 2.6, 0, 0, -5.0 * (kin.angV - kin.angE) * (3.15 / 1.68), 0.60, 0.60, 0.55, MATS.copper, 7);
    addPart("unitCyl", vPinX, vPinY, zArm + 2.6, 0, 0, -5.0 * (kin.angV - kin.angE) * (3.15 / 1.68), 0.36, 0.36, 0.90, MATS.ruby, 7);
    addPart("unitSphere", vx, vy, zOrb, 0, 0, -kin.angV * 0.9, 3.5, 3.5, 3.5, MATS.venus, 7);
    addPickSphere(vx, vy, zOrb, 6.5, 7);
    addOrbitRibbon(cx, cy, ECLIPTIC_PLANE_Z + exp * 32.0, R_V, 0.40, incV, 0.6, [0.90, 0.74, 0.44], lightTheme ? 0.36 : 0.30, 80);
    addVelocityTailArc(cx, cy, ECLIPTIC_PLANE_Z + exp * 32.0, R_V, 1.05, kin.angV - 0.6, 0.72, incV, 0.6, [1.0, 0.88, 0.56], lightTheme ? 0.52 : 0.68, 24);
  }

  // ========================================================================
  // GROUP 8: Earth Tellurian (5-Gear Parallel Train) & Antikythera Pin-and-Slot Lunar Epicycle
  // ========================================================================
  {
    const R_E = 45.0, zArm = zCollarE;
    const zEclE = ECLIPTIC_PLANE_Z + exp * 28.0;
    const dzE = zArm - 55.0;
    const ux = Math.cos(kin.angE), uy = Math.sin(kin.angE);
    const ex = cx + ux * R_E;
    const ey = cy + uy * R_E;
    const bx = cx - ux * 14.0;
    const by = cy - uy * 14.0;

    addPart("carrier_E", cx, cy, zArm, 0, 0, kin.angE, 1, 1, 1, MATS.brassHi, 8);
    addPart("unitCyl", bx, by, zArm + 3.2, 0, 0, 0, 1.1, 1.1, 5.2, MATS.steel, 8);
    addPart("foliotWeight", bx, by, zArm + 4.1, 0, 0, kin.angE, 0.85, 0.85, 0.80, MATS.copper, 8);
    addPart("daVinciWingNut", bx, by, zArm + 5.2, 0, 0, kin.angE, 0.52, 0.52, 0.52, MATS.bluedSteel, 8);
    addPart("unitSphere", bx, by, zArm + 6.5, 0, 0, 0, 2.35, 2.35, 2.35, MATS.brassHi, 8);
    addSegmentRod(
      [cx + ux * 4.1, cy + uy * 4.1, zArm + 1.3],
      [cx + ux * 39.5, cy + uy * 39.5, zArm + 0.8],
      0.24,
      MATS.bluedSteel,
      8
    );

    // True Epicyclic Zero-Slip 5-Gear Parallel-Linkage Train underneath Earth Arm (z = 52.4 + dzE):
    const STATIONS = [11.25, 22.50, 33.75, 45.00];
    for (let idx = 0; idx < STATIONS.length; idx++) {
      const sDist = STATIONS[idx];
      const gx = cx + ux * sDist;
      const gy = cy + uy * sDist;
      const isLast = idx === STATIONS.length - 1;
      const gRot = idx % 2 === 0 ? 2.0 * kin.angE + Math.PI / 24 : 0.0;
      const gMat = idx % 2 === 0 ? MATS.brass : MATS.copper;
      if (!isLast) {
        addPart("unitCyl", gx, gy, zArm - 1.4, 0, 0, 0, 1.25, 1.25, 2.6, MATS.bluedSteel, 8);
      }
      addPart("gear_tellurian", gx, gy, 52.4 + dzE, 0, 0, gRot, 1, 1, 1, gMat, 8);
    }

    // Stepped Earth Riser Column: lower journal neck (r = 0.45 at z = 56.2..58.7 + dzE clearing Group 9 marsPinion) + main shaft (r = 1.30 at tz = 68.5 + dzE) + upper neck (z = 78.3..81.1 + dzE)
    addPart("unitCyl", ex, ey, 57.45 + dzE, 0, 0, -kin.angMars * (1.030 / 1.470), 0.45, 0.45, 2.5, MATS.steel, 8);
    addPart("unitCyl", ex, ey, 68.5 + dzE, 0, 0, 0, 1.30, 1.30, 19.6, MATS.steel, 8);
    addPart("unitCyl", ex, ey, 79.7 + dzE, 0, 0, 0, 1.30, 1.30, 2.8, MATS.steel, 8);
    addPart("balusterPillar", ex, ey, zArm + 1.2, 0, 0, 0, 1.02, 1.02, 0.95, MATS.brassHi, 8);

    // Vertical Lunar Auxiliary Jackshaft (z = 52.4..73.2 + dzE) on perpendicular outrigger [jxLunar, jyLunar]:
    const jxLunar = ex - uy * 7.20;
    const jyLunar = ey + ux * 7.20;
    addSegmentRod([ex, ey, zArm], [jxLunar, jyLunar, zArm], 0.62, MATS.brassLo, 8);
    addSegmentRod([ex, ey, 72.1 + dzE], [jxLunar, jyLunar, 72.1 + dzE], 0.44, MATS.brassLo, 8);
    addPart("unitCyl", jxLunar, jyLunar, 62.8 + dzE, 0, 0, -kin.angE * 3.6, 0.44, 0.44, 21.2, MATS.bluedSteel, 8);
    addPart("gear_gal_io", jxLunar, jyLunar, 52.4 + dzE, 0, 0, -kin.angE * 3.6, 0.75, 0.75, 0.60, MATS.brassHi, 8);
    addPart("unitTube", jxLunar, jyLunar, 71.2 + dzE, 0, 0, -kin.angNodal * 0.8, 0.60, 0.60, 1.0, MATS.brassLo, 8);
    addPart("gear_gal_ca", jxLunar, jyLunar, 71.2 + dzE, 0, 0, -kin.angNodal * 0.8, 0.9524, 0.9524, 0.55, MATS.steel, 8);
    addPart("unitTube", jxLunar, jyLunar, 73.2 + dzE, 0, 0, -kin.apsidalAng * (4.40 / 2.80), 0.64, 0.64, 0.7, MATS.brassLo, 8);
    addPart("gear_gal_eu", jxLunar, jyLunar, 73.2 + dzE, 0, 0, -kin.apsidalAng * (4.40 / 2.80), 1.00, 1.00, 0.36, MATS.brassHi, 8);

    // Antikythera Fragment C Concentric-to-Eccentric Hipparchus Lunar Anomaly Stack (e5 / k1 / k2):
    // 1) Concentric 24T Apsidal Carrier Gear (e5, rPitch = 4.40) at [ex, ey, 73.2 + dzE] meshing at exact invariant 7.2000 with jackUpper!
    addPart("gear_jupHub", ex, ey, 73.2 + dzE, 0, 0, kin.apsidalAng, 4.4 / 4.8, 4.4 / 4.8, 0.24, MATS.brassHi, 8);
    // 2) Eccentric k1 Lunar Hub Gear at revolving apsidal offset e = 0.58 coupled to e5 via eccentric carrier journal & planetary drive pinion
    const eccX = ex + 0.58 * Math.cos(kin.apsidalAng);
    const eccY = ey + 0.58 * Math.sin(kin.apsidalAng);
    const k1PinX = eccX - Math.cos(kin.apsidalAng) * (4.40 + 1.26);
    const k1PinY = eccY - Math.sin(kin.apsidalAng) * (4.40 + 1.26);
    addSegmentRod([eccX, eccY, 73.45 + dzE], [k1PinX, k1PinY, 73.45 + dzE], 0.24, MATS.brassLo, 8);
    addPart("unitCyl", k1PinX, k1PinY, 73.70 + dzE, 0, 0, -kin.angMoonMean * (4.40 / 1.26), 0.34, 0.34, 0.65, MATS.bluedSteel, 8);
    addPart("gear_gal_io", k1PinX, k1PinY, 73.85 + dzE, 0, 0, -kin.angMoonMean * (4.40 / 1.26), 0.60, 0.60, 0.26, MATS.brassHi, 8);
    addPart("unitCyl", eccX, eccY, 73.60 + dzE, 0, 0, kin.apsidalAng, 2.02, 2.02, 0.25, MATS.brassLo, 8);
    addPart("gear_lunarHub", eccX, eccY, 73.85 + dzE, 0, 0, kin.angMoonMean, 1, 1, 0.26, MATS.copper, 8);
    // 3) Ruby impulse pin (r = 0.52) at rPin = 5.0 engaging k2 slotted follower arm (antikytheraSlotArm) & upper Hipparchus rim
    const pinX = eccX + Math.cos(kin.angMoonMean) * 5.0;
    const pinY = eccY + Math.sin(kin.angMoonMean) * 5.0;
    addPart("unitCyl", pinX, pinY, 74.75 + dzE, 0, 0, 0, 0.52, 0.52, 1.15, MATS.ruby, 8);
    addPart("antikytheraSlotArm", ex, ey, 75.2 + dzE, 0, 0, kin.angMoon, 1, 1, 1, MATS.brassHi, 8);
    addPart("jupMoonRing", ex, ey, 75.2 + dzE, 0, 0, kin.angMoon, 0.58, 0.58, 0.55, MATS.copper, 8);

    // Supported 5.14° Inclined Draconic Nodal Ring (z = 77.6 + dzE) with Outer 3-Arm Gimbal Spider (r = 11.65 > R_MOON = 9.8)
    // Driven by concentric toothed Draconic hub gear (gear_jup_ca, rPitch = 3.20) at [ex, ey, 71.2 + dzE] meshing with jackMid (rPitch = 4.00)
    const lunarInc = (5.14 * Math.PI) / 180;
    const ringRy = Math.asin(Math.sin(kin.angNodal) * Math.sin(lunarInc));
    const ringRx = Math.atan2(Math.cos(kin.angNodal) * Math.sin(lunarInc), Math.cos(lunarInc));
    // Exact composeTRS(rx, ry, rz) normal vector n = [sin(ry), -sin(rx)*cos(ry), cos(rx)*cos(ry)] gives plane height:
    const zRingAt = (dx, dy) =>
      77.6 + dzE + (-Math.tan(ringRy) * dx + (Math.tan(ringRx) / Math.cos(ringRy)) * dy);

    addPart("gear_jup_ca", ex, ey, 71.2 + dzE, 0, 0, kin.angNodal, 3.20 / 4.20, 3.20 / 4.20, 0.55, MATS.greekBronze, 8);
    for (let k = 0; k < 3; k++) {
      const sAng = kin.angNodal + (k * Math.PI * 2) / 3;
      const cs = Math.cos(sAng), sn = Math.sin(sAng);
      const pHub = [ex + cs * 1.85, ey + sn * 1.85, 72.3 + dzE];
      const pElbow = [ex + cs * 11.65, ey + sn * 11.65, 72.3 + dzE];
      const zRingRim = zRingAt(cs * 9.8, sn * 9.8);
      const pPostTop = [ex + cs * 11.65, ey + sn * 11.65, zRingRim];
      const pRingClamp = [ex + cs * 9.8, ey + sn * 9.8, zRingRim];
      addSegmentRod(pHub, pElbow, 0.34, MATS.brassLo, 8);
      addPart("unitCyl", pElbow[0], pElbow[1], pElbow[2], 0, 0, 0, 0.42, 0.42, 0.6, MATS.brassHi, 8);
      addSegmentRod(pElbow, pPostTop, 0.34, MATS.steel, 8);
      addSegmentRod(pPostTop, pRingClamp, 0.34, MATS.brassHi, 8);
      addPart("unitCyl", pRingClamp[0], pRingClamp[1], pRingClamp[2], ringRx, ringRy, sAng, 0.62, 0.62, 0.72, MATS.brassHi, 8);
    }
    addPart("lunarNodalRing", ex, ey, 77.6 + dzE, ringRx, ringRy, kin.angNodal, 1, 1, 1, MATS.copper, 8);

    // Lunar Carrier Arm, Forked Twin-Cheek Nodal Guide Stirrup (flanking the central Synodic Differential shaft!), & Moon Globe (R_MOON = 9.8)
    const R_MOON = 9.8;
    const mCos = Math.cos(kin.angMoon), mSin = Math.sin(kin.angMoon);
    const moonX = ex + mCos * R_MOON;
    const moonY = ey + mSin * R_MOON;
    const camZ = zRingAt(moonX - ex, moonY - ey);
    const moonZ = zEclE + (camZ - (77.6 + dzE));

    addPart("carrier_Moon", ex, ey, 76.2 + dzE, 0, 0, kin.angMoon, 1, 1, 1, MATS.brass, 8);
    // Twin lateral guide posts at r = 6.9, lateral offset +-1.15 so the central horizontal Synodic Differential shaft (y_local = 0) passes cleanly between them!
    for (const sgn of [-1, 1]) {
      const gx = ex + mCos * 6.9 - mSin * (sgn * 1.15);
      const gy = ey + mSin * 6.9 + mCos * (sgn * 1.15);
      addSegmentRod([gx, gy, 76.2 + dzE], [gx, gy, camZ + 0.55], 0.30, MATS.steel, 8);
      addSegmentRod([gx, gy, camZ + 0.55], [moonX, moonY, camZ + 0.55], 0.30, MATS.brassHi, 8);
    }
    addPart("unitCyl", moonX, moonY, camZ + 0.55, 0, 0, 0, 1.25, 1.25, 0.7, MATS.brassHi, 8);
    const mRiserBot = camZ + 0.55;
    const mRiserTop = moonZ - 1.75;
    const mRiserH = Math.max(1.0, mRiserTop - mRiserBot);
    addPart("unitCyl", moonX, moonY, mRiserBot + mRiserH * 0.5, 0, 0, 0, 0.68, 0.68, mRiserH, MATS.steel, 8);
    addPart("unitSphere", moonX, moonY, moonZ, 0, 0, kin.angMoon, 1.75, 1.75, 1.75, MATS.moon, 8);

    const moonAntiSunAz = Math.atan2(moonY - SUN_POS[1], moonX - SUN_POS[0]);
    addPart("lunarPhaseCup", moonX, moonY, moonZ, 0, 0, moonAntiSunAz, 1, 1, 1, MATS.bluedSteel, 8);

    // Closed-Loop Synodic Lunar Phase Differential Train (subtracting Earth orbital rate ω_E from sidereal lunar rate ω_sid):
    // 1) Central reference bevel gear at [ex, ey, 76.95 + dzE] meshing with inner miter bevel on horizontal differential shaft along carrier_Moon
    // Intrinsic composeTRS(rx, ry, rz) maps local +Z to [sin(ry), -sin(rx)*cos(ry), cos(rx)*cos(ry)] === [+-mCos, +-mSin, 0] with dot === 1.000000!
    const zSynShaft = 76.95 + dzE;
    const synInRx = Math.atan2(mSin, 0);
    const synInRy = Math.asin(Math.max(-1, Math.min(1, -mCos)));
    const synOutRx = Math.atan2(-mSin, 0);
    const synOutRy = Math.asin(Math.max(-1, Math.min(1, mCos)));
    addPart("bevelGear", ex, ey, zSynShaft - 0.55, 0, 0, -kin.angE, 0.42, 0.42, 0.42, MATS.copper, 8);
    addPart("bevelGear", ex + mCos * 1.95, ey + mSin * 1.95, zSynShaft, synInRx, synInRy, kin.relMoon * 2.0, 0.28, 0.28, 0.28, MATS.brassHi, 8);
    addSegmentRod(
      [ex + mCos * 1.95, ey + mSin * 1.95, zSynShaft],
      [ex + mCos * 8.65, ey + mSin * 8.65, zSynShaft],
      0.24,
      MATS.steel,
      8
    );
    addPart("unitCyl", ex + mCos * 5.2, ey + mSin * 5.2, zSynShaft - 0.35, 0, 0, kin.angMoon, 0.48, 0.48, 0.75, MATS.brassLo, 8);
    addPart("bevelGear", ex + mCos * 8.65, ey + mSin * 8.65, zSynShaft, synOutRx, synOutRy, -kin.relMoon * 2.0, 0.28, 0.28, 0.28, MATS.copper, 8);
    // 2) Lower & upper driven miter bevel pinions coaxial with [moonX, moonY] bracketing the outer synodic bevel at zSynShaft +- 0.55
    // & coaxial tubular phase sleeve coupling Moon riser up to lunarPhaseCup
    addPart("bevelGear", moonX, moonY, zSynShaft + 0.55, Math.PI, 0, kin.relMoon * 2.0, 0.28, 0.28, 0.28, MATS.brassHi, 8);
    addPart("bevelGear", moonX, moonY, zSynShaft - 0.55, 0, 0, kin.relMoon * 2.0, 0.32, 0.32, 0.32, MATS.brassHi, 8);
    addPart("unitTube", moonX, moonY, mRiserBot + mRiserH * 0.5, 0, 0, moonAntiSunAz, 0.88, 0.88, mRiserH * 0.82, MATS.brassHi, 8);

    // Earth Tellurian Horizon Ring, YZ Meridian Gimbal, Pedestal Cradle & Polar Axis (Always Tilted 23.44° toward +Y Celestial North!)
    const obliqRad = (23.44 * Math.PI) / 180;
    const earthSpin = (kin.day * 0.35) % (Math.PI * 2);
    // Turned brass pedestal cradle connecting Earth riser shaft top to earthMeridianRing bottom (z = zEclE - 4.95 = 104.05):
    addPart("unitCyl", ex, ey, zEclE - 5.15, 0, 0, 0, 1.85, 1.85, 1.1, MATS.brassHi, 8);
    addPart("unitCyl", ex, ey, zEclE - 4.65, 0, 0, 0, 1.35, 1.35, 0.7, MATS.copper, 8);
    // Earth Obliquity Orthogonal Miter-Bevel Diurnal Spin Drive on pedestal cradle shoulder meshing at 90° with tilted polar bevel pinion!
    const obApexY = ey - Math.sin(obliqRad) * 5.00;
    const obApexZ = zEclE - Math.cos(obliqRad) * 5.00;
    const obCrankY = obApexY + Math.cos(obliqRad) * 0.65;
    const obCrankZ = obApexZ + Math.sin(obliqRad) * 0.65;
    addSegmentRod([ex, ey, zEclE - 4.65], [ex, obCrankY, obCrankZ], 0.28, MATS.brassLo, 8);
    addPart("bevelGear", ex, obCrankY, obCrankZ, -obliqRad + Math.PI * 0.5, 0, earthSpin, 0.32, 0.32, 0.34, MATS.brassHi, 8);
    addPart(
      "bevelGear",
      ex,
      ey - Math.sin(obliqRad) * 4.35,
      zEclE - Math.cos(obliqRad) * 4.35,
      -obliqRad + Math.PI,
      0,
      -earthSpin,
      0.32,
      0.32,
      0.34,
      MATS.copper,
      8
    );
    addPart("earthHorizonRing", ex, ey, zEclE, 0, 0, 0, 1, 1, 1, MATS.brass, 8);
    addPart("earthMeridianRing", ex, ey, zEclE, 0, Math.PI * 0.5, 0, 1, 1, 1, MATS.brassHi, 8);
    // Horizontal gimbal trunnion pins joining earthMeridianRing (r = 4.95) to earthHorizonRing (r = 5.45) at ±Y:
    for (const sgn of [-1, 1]) {
      addSegmentRod([ex, ey + sgn * 4.75, zEclE], [ex, ey + sgn * 5.65, zEclE], 0.28, MATS.steel, 8);
      addPart("unitCyl", ex, ey + sgn * 5.20, zEclE, Math.PI * 0.5, 0, 0, 0.55, 0.55, 0.65, MATS.brassHi, 8);
    }
    // Polar axis rod trimmed cleanly to earthMeridianRing radius (r = 4.95) with coaxial tilted journal bosses:
    const pBot = [ex, ey - Math.sin(obliqRad) * 5.05, zEclE - Math.cos(obliqRad) * 5.05];
    const pTop = [ex, ey + Math.sin(obliqRad) * 5.05, zEclE + Math.cos(obliqRad) * 5.05];
    addSegmentRod(pBot, pTop, 0.44, MATS.chrome, 8);
    for (const sgn of [-1, 1]) {
      const jxP = ex;
      const jyP = ey + sgn * Math.sin(obliqRad) * 4.95;
      const jzP = zEclE + sgn * Math.cos(obliqRad) * 4.95;
      addPart("unitCyl", jxP, jyP, jzP, -obliqRad, 0, 0, 0.72, 0.72, 0.75, MATS.brassHi, 8);
    }
    addPart("unitCyl", pTop[0], pTop[1], pTop[2], -obliqRad, 0, 0, 0.48, 0.48, 0.35, MATS.ruby, 8);
    addPart("unitSphere", ex, ey, zEclE, -obliqRad, 0, earthSpin, 4.2, 4.2, 4.2, MATS.earth, 8);

    earthPos = [ex, ey, zEclE];
    moonPos = [moonX, moonY, moonZ];

    addPickSphere(ex, ey, zEclE, 9.0, 8);
    addPickSphere(moonX, moonY, moonZ, 4.5, 8);
    addPickSphere(cx + ux * 22.5, cy + uy * 22.5, zArm, 7.5, 8);

    addOrbitRibbon(cx, cy, zEclE, R_E, 0.52, 0, 0, [0.28, 0.66, 0.98], lightTheme ? 0.44 : 0.40, 96);
    addVelocityTailArc(cx, cy, zEclE, R_E, 1.25, kin.angE, 0.65, 0, 0, [0.42, 0.82, 1.0], lightTheme ? 0.58 : 0.74, 28);

    // 3D Eclipse Shadow Cones
    const sVecX = ex - SUN_POS[0], sVecY = ey - SUN_POS[1], sVecZ = zEclE - sunZ;
    const sDist = Math.hypot(sVecX, sVecY, sVecZ) || 1;
    const sDir = [sVecX / sDist, sVecY / sDist, sVecZ / sDist];
    const earthShadowEnd = [ex + sDir[0] * 14.0, ey + sDir[1] * 14.0, zEclE + sDir[2] * 14.0];
    addConeFrustumOverlay([ex, ey, zEclE], 4.0, earthShadowEnd, 1.8, [0.12, 0.16, 0.28], lightTheme ? 0.24 : 0.34, 24);

    const mVecX = moonX - SUN_POS[0], mVecY = moonY - SUN_POS[1], mVecZ = moonZ - sunZ;
    const mDist = Math.hypot(mVecX, mVecY, mVecZ) || 1;
    const mDir = [mVecX / mDist, mVecY / mDist, mVecZ / mDist];
    const moonShadowEnd = [moonX + mDir[0] * 10.5, moonY + mDir[1] * 10.5, moonZ + mDir[2] * 10.5];
    addConeFrustumOverlay([moonX, moonY, moonZ], 1.7, moonShadowEnd, 0.4, [0.18, 0.14, 0.22], lightTheme ? 0.22 : 0.32, 20);
  }

  // ========================================================================
  // GROUP 9: Mars Stepped Carrier Arm (z = 73.0, R = 62.0) & Local 1:18 Geocentric Epicycle Sighting Alidade
  // ========================================================================
  {
    const R_MARS = 62.0, zArm = zCollarMars;
    const incMars = (1.85 * Math.PI) / 180;
    const zOrb = ECLIPTIC_PLANE_Z + exp * 24.0 + Math.sin(kin.angMars - 0.8) * R_MARS * Math.sin(incMars);
    const mx = cx + Math.cos(kin.angMars) * R_MARS;
    const my = cy + Math.sin(kin.angMars) * R_MARS;
    const bx = cx - Math.cos(kin.angMars) * 15.5;
    const by = cy - Math.sin(kin.angMars) * 15.5;

    addPart("carrier_Mars", cx, cy, zArm, 0, 0, kin.angMars, 1, 1, 1, MATS.brass, 9);
    addPart("unitCyl", bx, by, zArm + 3.4, 0, 0, 0, 1.1, 1.1, 5.6, MATS.steel, 9);
    addPart("foliotWeight", bx, by, zArm + 4.4, 0, 0, kin.angMars, 0.92, 0.92, 0.85, MATS.brassHi, 9);
    addPart("daVinciWingNut", bx, by, zArm + 5.6, 0, 0, kin.angMars, 0.56, 0.56, 0.56, MATS.bluedSteel, 9);
    addPart("unitSphere", bx, by, zArm + 7.0, 0, 0, 0, 2.4, 2.4, 2.4, MATS.copper, 9);
    addSegmentRod(
      [cx + Math.cos(kin.angMars) * 5.0, cy + Math.sin(kin.angMars) * 5.0, zArm + 1.3],
      [cx + Math.cos(kin.angMars) * 54.0, cy + Math.sin(kin.angMars) * 54.0, zArm + 0.8],
      0.24,
      MATS.bluedSteel,
      9
    );

    const rH = zOrb - 3.1 - (zArm + 1.4);
    addPart("unitCyl", mx, my, zArm + 1.4 + rH * 0.5, 0, 0, 0, 1.22, 1.22, rH, MATS.steel, 9);
    addPart("balusterPillar", mx, my, zArm + 1.4, 0, 0, 0, 0.96, 0.96, 1.0, MATS.copper, 9);
    const marsObliq = (25.19 * Math.PI) / 180;
    addPart("unitSphere", mx, my, zOrb, -marsObliq, 0, kin.angMars * 3.0, 3.1, 3.1, 3.1, MATS.mars, 9);

    // Local 1:18 Geocentric Epicycle Resolver & Sighting Alidade on Earth Tellurian (z = zCollarE + 3.4 = 80.9)
    // Calibrated to 1:18 scale (max reach <= 6.26 < 6.62) so it sweeps 360° strictly inside the r = 7.20 Lunar Jackshaft!
    // Vector math: deferent hub at [ex, ey] - (R_E / 18) * u_E, epicycle crank radius (R_MARS / 18) at angle angMars
    // -> pin position relative to [ex, ey] is (1/18) * (r_Mars - r_Earth), 100% collinear with uEM!
    const zDioptra = zCollarE + 3.4;
    const ex = earthPos[0], ey = earthPos[1];
    const dxEM = mx - ex, dyEM = my - ey;
    const distEM = Math.hypot(dxEM, dyEM) || 1.0;
    const uxEM = dxEM / distEM, uyEM = dyEM / distEM;

    const defR = 45.0 / 18.0; // 2.500 units inward toward Sun along Earth carrier arm
    const epiR = R_MARS / 18.0; // 3.4444 units Mars epicycle radius
    const defX = ex - Math.cos(kin.angE) * defR;
    const defY = ey - Math.sin(kin.angE) * defR;
    const epiPinX = defX + Math.cos(kin.angMars) * epiR;
    const epiPinY = defY + Math.sin(kin.angMars) * epiR;

    // Deferent bracket, intermediate Tellurian drive pinion at [ex, ey] (rPitch = 2.10*0.70 = 1.470) meshing at exact
    // center distance D = 2.500 with Mars-rate epicycle wheel at [defX, defY] (rPitch = 3.50*0.294286 = 1.030 -> 1.470 + 1.030 = 2.500!):
    addPart("gear_gal_io", ex, ey, zDioptra - 0.65, 0, 0, -kin.angMars * (1.030 / 1.470), 0.70, 0.70, 0.85, MATS.brassHi, 9);
    addPart("unitCyl", defX, defY, zDioptra - 1.2, 0, 0, kin.angMars, 0.20, 0.20, 2.2, MATS.bluedSteel, 9);
    addPart("gear_gal_ga", defX, defY, zDioptra - 0.65, 0, 0, kin.angMars, 0.294286, 0.294286, 0.85, MATS.copper, 9);
    addPart("carrier_gal_io", defX, defY, zDioptra - 0.25, 0, 0, kin.angMars, 0.5065, 0.5065, 0.85, MATS.brassHi, 9);
    addPart("unitCyl", epiPinX, epiPinY, zDioptra + 0.15, 0, 0, 0, 0.34, 0.34, 1.1, MATS.ruby, 9);

    // Coaxial Dioptra azimuth collar & physical slotted follower yoke (slot [2.4*0.72=1.73, 7.6*0.72=5.47, rOuter=6.19 < 6.62]!)
    addPart("unitTube", ex, ey, zDioptra, 0, 0, kin.marsGeoLon, 2.15, 2.15, 1.2, MATS.brassHi, 9);
    addPart("antikytheraSlotArm", ex, ey, zDioptra + 0.25, 0, 0, kin.marsGeoLon, 0.72, 0.72, 0.65, MATS.brassHi, 9);
    addPart("unitTube", mx, my, zDioptra, 0, 0, kin.marsGeoLon, 2.05, 2.05, 1.2, MATS.copper, 9);

    // Compact 3-stage telescoping sighting alidade vane (max reach <= 6.11 units from Earth, clearing r = 7.20 Lunar Jackshaft & > 38 units from center!):
    const alidadeLen = Math.min(6.10, Math.max(3.6, distEM / 17.5));
    addSegmentRod(
      [ex + uxEM * 1.6, ey + uyEM * 1.6, zDioptra],
      [ex + uxEM * (alidadeLen * 0.62), ey + uyEM * (alidadeLen * 0.62), zDioptra],
      0.42,
      MATS.brassHi,
      9,
      true
    );
    addSegmentRod(
      [ex + uxEM * (alidadeLen * 0.30), ey + uyEM * (alidadeLen * 0.30), zDioptra],
      [ex + uxEM * (alidadeLen * 0.84), ey + uyEM * (alidadeLen * 0.84), zDioptra],
      0.32,
      MATS.copper,
      9,
      true
    );
    addSegmentRod(
      [ex + uxEM * (alidadeLen * 0.52), ey + uyEM * (alidadeLen * 0.52), zDioptra],
      [ex + uxEM * alidadeLen, ey + uyEM * alidadeLen, zDioptra],
      0.24,
      MATS.chrome,
      9,
      true
    );
    // Front & rear dioptra sighting pinnules + retrograde dragon pointer on Earth alidade tip & Mars target collar:
    const tipX = ex + uxEM * alidadeLen, tipY = ey + uyEM * alidadeLen;
    addPart("unitCyl", tipX, tipY, zDioptra + 0.45, 0, 0, kin.marsGeoLon, 0.44, 0.44, 0.9, MATS.brassHi, 9);
    addPart(
      "hand_dragon",
      tipX,
      tipY,
      zDioptra + 0.85,
      0,
      0,
      kin.marsGeoLon,
      0.28,
      0.28,
      0.52,
      kin.marsRetrograde ? MATS.ruby : MATS.bluedSteel,
      9
    );
    addPart(
      "hand_dragon",
      mx,
      my,
      zDioptra + 0.85,
      0,
      0,
      kin.marsGeoLon + Math.PI,
      0.34,
      0.34,
      0.52,
      kin.marsRetrograde ? MATS.ruby : MATS.bluedSteel,
      9
    );

    // Optical Geocentric Sightline Ray Overlay from Earth Alidade Tip to Mars Collar + Retrograde Loop Ribbon on Celestial Vault
    addConeFrustumOverlay(
      [tipX, tipY, zDioptra],
      0.16,
      [mx - uxEM * 2.0, my - uyEM * 2.0, zDioptra],
      0.16,
      kin.marsRetrograde ? [1.0, 0.32, 0.22] : [0.95, 0.62, 0.30],
      lightTheme ? 0.34 : 0.48,
      8
    );

    const R_VAULT = 95.0;
    let prevPt = null;
    const retroSteps = 36;
    for (let s = 0; s <= retroSteps; s++) {
      const dSample = kin.day - 90.0 + (s / retroSteps) * 180.0;
      const aES = (dSample / T_EARTH) * Math.PI * 2;
      const aMS = (dSample / T_MARS) * Math.PI * 2 + 1.92;
      const exS = cx + Math.cos(aES) * 45.0, eyS = cy + Math.sin(aES) * 45.0;
      const mxS = cx + Math.cos(aMS) * R_MARS, myS = cy + Math.sin(aMS) * R_MARS;
      const gAz = Math.atan2(myS - eyS, mxS - exS);
      const zS = ECLIPTIC_PLANE_Z + exp * 24.0 + Math.sin((s / retroSteps) * Math.PI * 2) * 2.2;
      const curPt = [cx + Math.cos(gAz) * R_VAULT, cy + Math.sin(gAz) * R_VAULT, zS];
      if (prevPt) {
        const wRib = 0.55;
        const fade = Math.sin((s / retroSteps) * Math.PI);
        const rCol = kin.marsRetrograde ? [1.0, 0.36, 0.24] : [0.95, 0.58, 0.30];
        const al = (lightTheme ? 0.32 : 0.46) * fade;
        pushOverlayVert(prevPt[0], prevPt[1], prevPt[2] - wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 0, 0);
        pushOverlayVert(prevPt[0], prevPt[1], prevPt[2] + wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 1, 0);
        pushOverlayVert(curPt[0], curPt[1], curPt[2] + wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 1, 0);

        pushOverlayVert(prevPt[0], prevPt[1], prevPt[2] - wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 0, 0);
        pushOverlayVert(curPt[0], curPt[1], curPt[2] + wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 1, 0);
        pushOverlayVert(curPt[0], curPt[1], curPt[2] - wRib, 0, 0, 1, rCol[0], rCol[1], rCol[2], al, 0, 0);
      }
      prevPt = curPt;
    }

    addPickSphere(mx, my, zOrb, 6.8, 9);
    addPickSphere(defX, defY, zDioptra, 4.5, 9);
    addOrbitRibbon(cx, cy, ECLIPTIC_PLANE_Z + exp * 24.0, R_MARS, 0.44, incMars, 0.8, [0.88, 0.44, 0.24], lightTheme ? 0.36 : 0.30, 96);
    addVelocityTailArc(cx, cy, ECLIPTIC_PLANE_Z + exp * 24.0, R_MARS, 1.15, kin.angMars - 0.8, 0.55, incMars, 0.8, [1.0, 0.52, 0.28], lightTheme ? 0.54 : 0.70, 24);
  }

  // ========================================================================
  // GROUP 15: Jupiter Carrier Arm (z = 43.5, R = 76.0) & 4-Tier Coaxial Galilean Tower
  // ========================================================================
  {
    const R_JUP = 76.0, zArm = zCollarJup;
    const incJup = (1.30 * Math.PI) / 180;
    const zOrb = ECLIPTIC_PLANE_Z + exp * 20.0 + Math.sin(kin.angJup + 0.4) * R_JUP * Math.sin(incJup);
    const jx = cx + Math.cos(kin.angJup) * R_JUP;
    const jy = cy + Math.sin(kin.angJup) * R_JUP;
    const bx = cx - Math.cos(kin.angJup) * 17.5;
    const by = cy - Math.sin(kin.angJup) * 17.5;

    addPart("carrier_Jup", cx, cy, zArm, 0, 0, kin.angJup, 1, 1, 1, MATS.copper, 15);
    addPart("unitCyl", bx, by, zArm + 3.6, 0, 0, 0, 1.2, 1.2, 6.2, MATS.steel, 15);
    addPart("foliotWeight", bx, by, zArm + 4.8, 0, 0, kin.angJup, 1.02, 1.02, 0.92, MATS.copper, 15);
    addPart("daVinciWingNut", bx, by, zArm + 6.1, 0, 0, kin.angJup, 0.62, 0.62, 0.62, MATS.bluedSteel, 15);
    addPart("unitSphere", bx, by, zArm + 7.6, 0, 0, 0, 2.75, 2.75, 2.75, MATS.brassHi, 15);
    addSegmentRod(
      [cx + Math.cos(kin.angJup) * 5.8, cy + Math.sin(kin.angJup) * 5.8, zArm + 1.4],
      [cx + Math.cos(kin.angJup) * 66.0, cy + Math.sin(kin.angJup) * 66.0, zArm + 0.9],
      0.26,
      MATS.bluedSteel,
      15
    );

    const zGal = zOrb - 10.5;
    const rH = (zGal - 3.4) - (zArm + 1.5);
    addPart("unitCyl", jx, jy, zArm + 1.5 + rH * 0.5, 0, 0, 0, 1.45, 1.45, rH, MATS.steel, 15);
    addPart("balusterPillar", jx, jy, zArm + 1.5, 0, 0, 0, 1.08, 1.08, 1.15, MATS.brassHi, 15);

    addPart("gear_jupHub", jx, jy, zGal - 2.65, 0, 0, kin.day * 0.85, 1, 1, 0.45, MATS.brassHi, 15);
    // Lowered jupMoonRing to zGal - 3.4 (below gear_jupHub and all 4 Galilean moon gears/carriers)
    // and anchored by 3 radial spider struts to the Jupiter riser hub so it never floats or blocks moon stalks:
    addPart("jupMoonRing", jx, jy, zGal - 3.4, 0, 0, kin.angJup, 1, 1, 1, MATS.brass, 15);
    for (let k = 0; k < 3; k++) {
      const sAng = kin.angJup + ((k * 120 + 60) * Math.PI) / 180;
      const cs = Math.cos(sAng), sn = Math.sin(sAng);
      addSegmentRod(
        [jx + cs * 1.45, jy + sn * 1.45, zGal - 3.4],
        [jx + cs * 10.9, jy + sn * 10.9, zGal - 3.4],
        0.28,
        MATS.brassLo,
        15,
        true
      );
    }

    const gcx = jx - Math.cos(kin.angJup) * 6.30;
    const gcy = jy - Math.sin(kin.angJup) * 6.30;
    addSegmentRod([jx, jy, zGal - 3.4], [gcx, gcy, zGal - 3.4], 0.42, MATS.brassLo, 15, true);
    // Upper countershaft bearing bridge at zGal + 2.05 (below lowest carrier_gal_ca at zGal + 2.4):
    addSegmentRod(
      [jx - Math.cos(kin.angJup) * 1.85, jy - Math.sin(kin.angJup) * 1.85, zGal + 2.05],
      [gcx, gcy, zGal + 2.05],
      0.34,
      MATS.brassHi,
      15,
      true
    );
    addPart("unitCyl", gcx, gcy, zGal - 0.55, 0, 0, -kin.day * 0.85 * (4.80 / 1.50), 0.45, 0.45, 5.6, MATS.bluedSteel, 15);
    addPart("gear_gal_io", gcx, gcy, zGal - 2.65, 0, 0, -kin.day * 0.85 * (4.80 / 1.50), 1.50 / 2.10, 1.50 / 2.10, 0.45, MATS.copper, 15);
    addPart("unitTube", gcx, gcy, zGal - 1.5, 0, 0, -kin.galIoAng * 0.5, 0.58, 0.58, 0.70, MATS.brassLo, 15);
    addPart("gear_gal_ca", gcx, gcy, zGal - 1.5, 0, 0, -kin.galIoAng * 0.5, 1.0, 1.0, 0.58, MATS.copper, 15);
    addPart("unitTube", gcx, gcy, zGal - 0.5, 0, 0, -kin.galEuAng * 0.8, 0.58, 0.58, 0.70, MATS.brassLo, 15);
    addPart("gear_gal_ga", gcx, gcy, zGal - 0.5, 0, 0, -kin.galEuAng * 0.8, 1.0, 1.0, 0.58, MATS.brassHi, 15);
    addPart("unitTube", gcx, gcy, zGal + 0.5, 0, 0, -kin.galGaAng * 1.25, 0.58, 0.58, 0.70, MATS.brassLo, 15);
    addPart("gear_gal_eu", gcx, gcy, zGal + 0.5, 0, 0, -kin.galGaAng * 1.25, 1.0, 1.0, 0.58, MATS.copper, 15);
    addPart("unitTube", gcx, gcy, zGal + 1.5, 0, 0, -kin.galCaAng * 2.0, 0.58, 0.58, 0.70, MATS.brassLo, 15);
    addPart("gear_gal_io", gcx, gcy, zGal + 1.5, 0, 0, -kin.galCaAng * 2.0, 1.0, 1.0, 0.58, MATS.brassHi, 15);

    const GALILEAN_TIERS = [
      {
        gearMesh: "gear_jup_io",
        carrierMesh: "carrier_gal_io",
        zGear: zGal - 1.5,
        zCarrier: zGal + 6.9,
        rSleeve: 0.72,
        rOrb: 6.8,
        ang: kin.galIoAng,
        rSphere: 0.82,
        gearMat: MATS.brassHi,
        carrierMat: MATS.brassHi,
        moonMat: MATS.brassHi,
      },
      {
        gearMesh: "gear_jup_eu",
        carrierMesh: "carrier_gal_eu",
        zGear: zGal - 0.5,
        zCarrier: zGal + 5.4,
        rSleeve: 0.98,
        rOrb: 8.2,
        ang: kin.galEuAng,
        rSphere: 0.76,
        gearMat: MATS.copper,
        carrierMat: MATS.brass,
        moonMat: MATS.moon,
      },
      {
        gearMesh: "gear_jup_ga",
        carrierMesh: "carrier_gal_ga",
        zGear: zGal + 0.5,
        zCarrier: zGal + 3.9,
        rSleeve: 1.32,
        rOrb: 9.8,
        ang: kin.galGaAng,
        rSphere: 1.02,
        gearMat: MATS.brassHi,
        carrierMat: MATS.copper,
        moonMat: MATS.mercury,
      },
      {
        gearMesh: "gear_jup_ca",
        carrierMesh: "carrier_gal_ca",
        zGear: zGal + 1.5,
        zCarrier: zGal + 2.4,
        rSleeve: 1.78,
        rOrb: 11.4,
        ang: kin.galCaAng,
        rSphere: 0.92,
        gearMat: MATS.copper,
        carrierMat: MATS.brassLo,
        moonMat: MATS.copper,
      },
    ];

    for (const gt of GALILEAN_TIERS) {
      const sH = gt.zCarrier - gt.zGear;
      addPart(gt.gearMesh, jx, jy, gt.zGear, 0, 0, gt.ang, 1, 1, 0.58, gt.gearMat, 15);
      addPart("unitTube", jx, jy, gt.zGear + sH * 0.5, 0, 0, gt.ang, gt.rSleeve, gt.rSleeve, sH, MATS.steel, 15);
      addPart(gt.carrierMesh, jx, jy, gt.zCarrier, 0, 0, gt.ang, 1, 1, 1, gt.carrierMat, 15);
      const gmx = jx + Math.cos(gt.ang) * gt.rOrb;
      const gmy = jy + Math.sin(gt.ang) * gt.rOrb;
      addSegmentRod([gmx, gmy, gt.zCarrier], [gmx, gmy, zOrb], 0.26, MATS.steel, 15, true);
      addPart("unitSphere", gmx, gmy, zOrb, 0, 0, gt.ang, gt.rSphere, gt.rSphere, gt.rSphere, gt.moonMat, 15);
    }

    addPart("unitCyl", jx, jy, zGal + 2.2, 0, 0, kin.day * 0.85, 0.52, 0.52, 11.2, MATS.chrome, 15);
    addPart("unitSphere", jx, jy, zOrb, -0.055, 0, kin.day * 0.85, 4.8, 4.8, 4.8, MATS.jupiter, 15);
    addPickSphere(jx, jy, zOrb, 9.5, 15);
    addOrbitRibbon(cx, cy, ECLIPTIC_PLANE_Z + exp * 20.0, R_JUP, 0.48, incJup, -0.4, [0.88, 0.70, 0.44], lightTheme ? 0.36 : 0.30, 112);
    addVelocityTailArc(cx, cy, ECLIPTIC_PLANE_Z + exp * 20.0, R_JUP, 1.30, kin.angJup + 0.4, 0.48, incJup, -0.4, [1.0, 0.82, 0.52], lightTheme ? 0.52 : 0.68, 24);
  }

  // ========================================================================
  // GROUP 16: Saturn Outer Carrier Arm (z = 38.5, R = 90.0), 26.73° Cassini Ring System & Titan
  // ========================================================================
  {
    const R_SAT = 90.0, zArm = zCollarSat;
    const incSat = (2.49 * Math.PI) / 180;
    const zOrb = ECLIPTIC_PLANE_Z + exp * 16.0 + Math.sin(kin.angSat - 1.1) * R_SAT * Math.sin(incSat);
    const sx = cx + Math.cos(kin.angSat) * R_SAT;
    const sy = cy + Math.sin(kin.angSat) * R_SAT;
    const bx = cx - Math.cos(kin.angSat) * 19.5;
    const by = cy - Math.sin(kin.angSat) * 19.5;

    addPart("carrier_Sat", cx, cy, zArm, 0, 0, kin.angSat, 1, 1, 1, MATS.brassHi, 16);
    addPart("unitCyl", bx, by, zArm + 3.8, 0, 0, 0, 1.25, 1.25, 6.4, MATS.steel, 16);
    addPart("foliotWeight", bx, by, zArm + 5.0, 0, 0, kin.angSat, 1.12, 1.12, 0.98, MATS.brassHi, 16);
    addPart("daVinciWingNut", bx, by, zArm + 6.4, 0, 0, kin.angSat, 0.66, 0.66, 0.66, MATS.bluedSteel, 16);
    addPart("unitSphere", bx, by, zArm + 7.9, 0, 0, 0, 2.95, 2.95, 2.95, MATS.copper, 16);
    addSegmentRod(
      [cx + Math.cos(kin.angSat) * 6.5, cy + Math.sin(kin.angSat) * 6.5, zArm + 1.5],
      [cx + Math.cos(kin.angSat) * 78.0, cy + Math.sin(kin.angSat) * 78.0, zArm + 0.9],
      0.28,
      MATS.bluedSteel,
      16
    );

    const rH = (zOrb - 5.55) - (zArm + 1.5);
    addPart("unitCyl", sx, sy, zArm + 1.5 + rH * 0.5, 0, 0, 0, 1.48, 1.48, rH, MATS.steel, 16);
    addPart("balusterPillar", sx, sy, zArm + 1.5, 0, 0, 0, 1.15, 1.15, 1.25, MATS.brassHi, 16);
    // Turned Meridian Gimbal Cradle at top of Saturn riser (z = zOrb - 5.55) supporting earthMeridianRing (r = 5.54):
    addPart("unitCyl", sx, sy, zOrb - 5.95, 0, 0, 0, 2.05, 2.05, 1.0, MATS.brassHi, 16);
    addPart("unitCyl", sx, sy, zOrb - 5.35, 0, 0, 0, 1.52, 1.52, 0.75, MATS.copper, 16);

    // Saturn YZ Meridian Gimbal Ring & 26.73° Tilted Polar Axis Rod (trimmed to r = 5.54 ring with coaxial journal bosses)
    const satObliq = (26.73 * Math.PI) / 180;
    const satSpin = (kin.day * 0.78) % (Math.PI * 2);
    addPart("earthMeridianRing", sx, sy, zOrb, 0, Math.PI * 0.5, 0, 1.12, 1.12, 1.12, MATS.brassHi, 16);
    const pBotSat = [sx, sy - Math.sin(satObliq) * 5.65, zOrb - Math.cos(satObliq) * 5.65];
    const pTopSat = [sx, sy + Math.sin(satObliq) * 5.65, zOrb + Math.cos(satObliq) * 5.65];
    addSegmentRod(pBotSat, pTopSat, 0.44, MATS.chrome, 16);
    for (const sgn of [-1, 1]) {
      addPart(
        "unitCyl",
        sx,
        sy + sgn * Math.sin(satObliq) * 5.54,
        zOrb + sgn * Math.cos(satObliq) * 5.54,
        -satObliq,
        0,
        0,
        0.76,
        0.76,
        0.8,
        MATS.brassHi,
        16
      );
    }

    // Banded Saturn Globe & Coaxial 26.73° Tilted Concentric Cassini Ring System
    addPart("unitSphere", sx, sy, zOrb, -satObliq, 0, satSpin, 4.25, 4.25, 4.25, MATS.saturn, 16);
    addPart("saturnRings", sx, sy, zOrb, -satObliq, 0, satSpin * 0.5, 1, 1, 1, MATS.saturnRings, 16);

    // Orbiting Titan Satellite (rOrb = 14.8) mounted via a tilted-ring-plane outrigger clamp on the outer rim of saturnRings (r = 12.2)
    // so the support arm never sweeps through the fixed vertical YZ earthMeridianRing!
    const rRingRim = 12.2;
    const rTitan = 14.8;
    const cTit = Math.cos(kin.titanAng), sTit = Math.sin(kin.titanAng);
    const rimX = sx + cTit * rRingRim;
    const rimY = sy + sTit * rRingRim * Math.cos(satObliq);
    const rimZ = zOrb - sTit * rRingRim * Math.sin(satObliq);
    const titanX = sx + cTit * rTitan;
    const titanY = sy + sTit * rTitan * Math.cos(satObliq);
    const titanZ = zOrb - sTit * rTitan * Math.sin(satObliq);
    addPart("unitCyl", rimX, rimY, rimZ, -satObliq, 0, kin.titanAng, 0.52, 0.52, 0.65, MATS.brassHi, 16);
    addSegmentRod([rimX, rimY, rimZ], [titanX, titanY, titanZ], 0.24, MATS.brassHi, 16);
    addPart("unitSphere", titanX, titanY, titanZ, 0, 0, kin.titanAng, 0.98, 0.98, 0.98, MATS.venus, 16);

    addPickSphere(sx, sy, zOrb, 12.5, 16);
    addOrbitRibbon(cx, cy, ECLIPTIC_PLANE_Z + exp * 16.0, R_SAT, 0.52, incSat, 1.1, [0.92, 0.78, 0.48], lightTheme ? 0.38 : 0.34, 128);
    addVelocityTailArc(cx, cy, ECLIPTIC_PLANE_Z + exp * 16.0, R_SAT, 1.38, kin.angSat - 1.1, 0.42, incSat, 1.1, [1.0, 0.88, 0.58], lightTheme ? 0.54 : 0.70, 28);
  }

  // ========================================================================
  // GROUP 10: 6-Column Greek Zodiac Parapegma, 3-Arm Spider Bearing Bridge & Centered Omega Celestial Armillary Sphere
  // ========================================================================
  {
    addPart("horizonRing", cx, cy, 5.05, 0, 0, 0, 1, 1, 1, MATS.brassLo, 10);
    const zEclLift = exp * 14.0;
    const szCol = 1.0 + zEclLift / 33.0;
    const PILLAR_DEGS = [-18, 25, 68, 112, 155, 198];
    const PILLAR_R = 48.0;
    for (const deg of PILLAR_DEGS) {
      const ang = (deg * Math.PI) / 180;
      const px = cx + Math.cos(ang) * PILLAR_R;
      const py = cy + Math.sin(ang) * PILLAR_R;
      addPart("greekColumn", px, py, 5.0, 0, 0, ang, 1, 1, szCol, MATS.greekBronze, 10);
      addPickSphere(px, py, 24.0, 8.5, 10);
    }
    addPart("eclipticRing", cx, cy, 38.0 + zEclLift, 0, 0, 0, 1, 1, 1, MATS.brassHi, 10);
    // Antikythera Parapegma Star-Almanac Index Pegs & Cardinal Ruby Jewels restricted strictly to the solid [-25°, +205°] arc of eclipticRing!
    for (let zIdx = 0; zIdx <= 6; zIdx++) {
      const zAng = (zIdx * Math.PI) / 6;
      const zCs = Math.cos(zAng), zSn = Math.sin(zAng);
      const pegRx = Math.atan2(-zSn, 0);
      const pegRy = Math.asin(Math.max(-1, Math.min(1, zCs)));
      addPart("unitCyl", cx + zCs * 52.35, cy + zSn * 52.35, 39.25 + zEclLift, pegRx, pegRy, 0, 0.30, 0.30, 0.75, MATS.brassHi, 10);
      if (zIdx % 3 === 0) {
        addPart("unitCyl", cx + zCs * 50.8, cy + zSn * 50.8, 40.55 + zEclLift, 0, 0, 0, 0.44, 0.44, 0.36, MATS.ruby, 10);
      }
    }
    // Terminal cutaway end-stop jewel chatons seated on the solid ends (-23° and +203°) of eclipticRing
    for (const endDeg of [-23, 203]) {
      const endAng = (endDeg * Math.PI) / 180;
      addPart("jewelChaton", cx + Math.cos(endAng) * 48.0, cy + Math.sin(endAng) * 48.0, 40.55 + zEclLift, 0, 0, endAng, 0.72, 0.72, 0.62, MATS.brassHi, 10);
      addPart("unitCyl", cx + Math.cos(endAng) * 48.0, cy + Math.sin(endAng) * 48.0, 40.68 + zEclLift, 0, 0, 0, 0.58, 0.58, 0.36, MATS.ruby, 10);
    }
    // Revolving Solar Ecliptic Fiducial Carriage (clamped to the solid [-22°, +202°] eclipticRing arc so it never floats in the front cutaway gap)
    const sunDegNorm = (((kin.angE * 180) / Math.PI) % 360 + 360) % 360;
    const sunFidAng =
      sunDegNorm <= 202 || sunDegNorm >= 338
        ? kin.angE
        : sunDegNorm < 270
        ? (202 * Math.PI) / 180
        : (-22 * Math.PI) / 180;
    const sunFidX = cx + Math.cos(sunFidAng) * 49.2;
    const sunFidY = cy + Math.sin(sunFidAng) * 49.2;
    addPart("hand_sun", sunFidX, sunFidY, 40.58 + zEclLift, 0, 0, sunFidAng + Math.PI, 0.22, 0.22, 0.45, MATS.copper, 10);
    addPart("unitCyl", sunFidX, sunFidY, 40.78 + zEclLift, 0, 0, 0, 0.48, 0.48, 0.38, MATS.ruby, 10);

    // 3-Arm Pierced Swiss Zodiac Spider Bearing Bridge ("Araignée de Centrage") at z = 39.8 + zEclLift with Annular Incabloc Lyre Spring (rIn = 6.99 > 6.70!) & Peripheral Chatons:
    addPart("unitTube", cx, cy, 39.8 + zEclLift, 0, 0, 0, 9.0, 9.0, 2.2, MATS.greekBronze, 10);
    addPart("lyreSpring", cx, cy, 41.05 + zEclLift, 0, 0, 0.45, 5.6, 5.6, 1.2, MATS.bluedSteel, 10);
    for (const deg of [25, 112, 198]) {
      const ang = (deg * Math.PI) / 180;
      const cs = Math.cos(ang), sn = Math.sin(ang);
      addPart("spiderBridgeArm", cx, cy, 39.8 + zEclLift, 0, 0, ang, 1, 1, 1, MATS.brassHi, 10);
      addPart("jewelChaton", cx + cs * 44.2, cy + sn * 44.2, 40.68 + zEclLift, 0, 0, ang, 0.85, 0.85, 0.72, MATS.brassHi, 10);
      addPart("slottedScrew", cx + cs * 44.2, cy + sn * 44.2, 40.82 + zEclLift, 0, 0, ang + 0.5, 0.92, 0.92, 0.95, MATS.bluedSteel, 10);
      addPart("slottedScrew", cx + cs * 8.1, cy + sn * 8.1, 41.02 + zEclLift, 0, 0, ang, 0.62, 0.62, 0.85, MATS.bluedSteel, 10);
    }

    // Centered 3-Ribbed Omega-Profile Celestial Meridian (68° & 112° over rear Corinthian columns) & Solstitial Colure (90°) Arches
    // anchored in turned sockets on the Ecliptic Ring (r = 48.0, z = 40.5 + zEclLift), flaring to r = 108.5 across the planetary zone,
    // and converging to the Open Zenith Polar Ring at [cx, cy, 126.0] directly above the Sun without obstructing front views or side turrets:
    for (const sockDeg of [68, 90, 112]) {
      const sAng = (sockDeg * Math.PI) / 180;
      const sxA = cx + Math.cos(sAng) * 48.0;
      const syA = cy + Math.sin(sAng) * 48.0;
      addPart("unitCyl", sxA, syA, 39.8 + zEclLift, 0, 0, sAng, 2.6, 2.6, 3.6, MATS.brassHi, 10);
      addPart("daVinciWingNut", sxA, syA, 41.55 + zEclLift, 0, 0, sAng, 0.72, 0.72, 0.72, MATS.brass, 10);
    }
    const zArmBase = 40.5 + zEclLift;
    const szArm = 1.0 + exp * (18.0 / 85.5);
    const zZenith = zArmBase + 85.5 * szArm;
    addPart("armillaryArch", cx, cy, zArmBase, 0, 0, (68 * Math.PI) / 180, 1.0, 1.0, szArm, MATS.greekBronze, 10);
    addPart("armillaryColure", cx, cy, zArmBase, 0, 0, (90 * Math.PI) / 180, 1.0, 1.0, szArm, MATS.brassHi, 10);
    addPart("armillaryArch", cx, cy, zArmBase, 0, 0, (112 * Math.PI) / 180, 1.0, 1.0, szArm, MATS.greekBronze, 10);

    // Equinoctial & Tropical Parallel Tie-Arcs bracing the 3 Armillary Ribs across 68°..112° in the rear hemisphere:
    for (const [rTie, zTieBase] of [
      [107.2, 101.0],
      [99.6, 120.8],
    ]) {
      const zTie = zArmBase + (zTieBase - 40.5) * szArm;
      const a68 = (68 * Math.PI) / 180;
      const a90 = (90 * Math.PI) / 180;
      const a112 = (112 * Math.PI) / 180;
      const p68 = [cx + Math.cos(a68) * rTie, cy + Math.sin(a68) * rTie, zTie];
      const p90 = [cx + Math.cos(a90) * rTie, cy + Math.sin(a90) * rTie, zTie];
      const p112 = [cx + Math.cos(a112) * rTie, cy + Math.sin(a112) * rTie, zTie];
      addSegmentRod(p68, p90, 0.42, MATS.brassHi, 10);
      addSegmentRod(p90, p112, 0.42, MATS.brassHi, 10);
      for (const pt of [p68, p90, p112]) {
        addPart("unitCyl", pt[0], pt[1], pt[2], 0, 0, 0, 1.05, 1.05, 1.4, MATS.brassHi, 10);
      }
    }

    // Open Zenith Polar Ring, Clamping Boss & Ruby Celestial Pole Finial at [cx, cy, zZenith]:
    addPart("unitTube", cx, cy, zZenith, 0, 0, 0, 3.4, 3.4, 1.8, MATS.brassHi, 10);
    addPart("unitCyl", cx, cy, zZenith + 1.4, 0, 0, 0, 0.55, 0.55, 2.8, MATS.brassHi, 10);
    addPart("earthMeridianRing", cx, cy, zZenith + 2.8, Math.PI * 0.5, 0, 0, 0.58, 0.58, 0.58, MATS.brassHi, 10);
    addPart("unitSphere", cx, cy, zZenith + 2.8, 0, 0, 0, 1.25, 1.25, 1.25, MATS.ruby, 10);
    addPickSphere(cx, cy + 48.0, 40.5, 14.0, 10);
    addPickSphere(cx, cy, zZenith, 10.0, 10);
  }

  // ========================================================================
  // GROUP 11: Harmonic Scotch-Yoke Solar Declination Gauge & Equation-of-Time Bell-Crank Cam
  // ========================================================================
  {
    addPart("ephemBox", 94, 0, 0, 0, 0, 0, 1, 1, 1, MATS.gunmetal, 11);
    for (const [bx, by, zSurf] of [
      [60.6, -29.4, 8.5],  [94.0, -29.4, 8.5],  [127.4, -29.4, 13.0],
      [60.6,  29.4, 13.0], [94.0,  29.4, 13.0], [127.4,  29.4, 13.0],
    ]) {
      addPart("unitCyl", bx, by, zSurf + 0.15, 0, 0, 0, 1.55, 1.55, 0.35, MATS.brassHi, 11);
      addPart("unitCyl", bx, by, zSurf + 0.42, 0, 0, 0, 1.0, 1.0, 0.45, MATS.bluedSteel, 11);
      addPart("daVinciWingNut", bx, by, zSurf + 0.58, 0, 0, -0.4, 0.65, 0.65, 0.65, MATS.brass, 11);
    }

    // Open 3-Branch Bevel & Spur Distribution Train seated in the Recessed Front Transmission Gallery of ephemBox (z = 8.5..13.0) receiving Right Line-Shaft at [65.4, -23.43]:
    addPart("bevelGear", 66.8, ly, 10.8, 0, -Math.PI * 0.5, -kin.angE * 4.0, 0.55, 0.55, 0.55, MATS.brassHi, 11);
    addPart("bevelGear", 65.4, ly, 12.2, Math.PI, 0, kin.angE, 0.55, 0.55, 0.55, MATS.copper, 11);
    addPart("unitCyl", 65.4, ly, 9.8, 0, 0, 0, 1.38, 1.38, 2.6, MATS.brassLo, 11);
    addPart("unitCyl", 65.4, ly, 10.8, 0, Math.PI * 0.5, 0, 1.28, 1.28, 1.8, MATS.brass, 11);
    addPart("unitCyl", 65.4, ly, 13.9, 0, 0, -kin.angE, 0.68, 0.68, 3.8, MATS.steel, 11);
    addPart("gear_gal_ca", 65.4, ly, 14.55, 0, 0, -kin.angE, 1.25, 1.25, 0.50, MATS.brassHi, 11);
    addPart("oilCup", 65.4, ly, 15.88, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 11);
    // Branch A: Collinear +X shaft continuation along y = ly = -23.43, z = 10.8 to Scotch-Yoke Declination Miter Bevel at x = 93.85..95.2 (100% outside declBarrel [95.2, -11.8]!)
    addSegmentRod([65.4, ly, 10.8], [93.85, ly, 10.8], 0.92, MATS.steel, 11);
    addPart("flangedCoupling", 79.5, ly, 10.8, 0, Math.PI * 0.5, kin.angE * 4.0, 0.72, 0.72, 0.72, MATS.brassHi, 11);
    addPart("unitCyl", 79.5, ly, 9.3, 0, 0, 0, 1.35, 1.35, 1.6, MATS.brassLo, 11);
    addPart("bevelGear", 93.85, ly, 10.8, 0, Math.PI * 0.5, kin.angE * 4.0, 0.52, 0.52, 0.52, MATS.copper, 11);
    addPart("bevelGear", 95.2, ly, 12.15, Math.PI, 0, -kin.angE * 2.0, 0.52, 0.52, 0.52, MATS.brassHi, 11);
    addPart("unitCyl", 95.2, ly, 9.8, 0, 0, 0, 1.38, 1.38, 2.6, MATS.brassLo, 11);
    addPart("unitCyl", 95.2, ly, 10.8, 0, Math.PI * 0.5, 0, 1.28, 1.28, 1.8, MATS.brass, 11);
    // Vertical Declination Drive Arbor [95.2, ly, 11.5..25.0] & Upper 90° Miter Bevel Pair at z = 22.80..24.0 with Outrigger Bearing Boss & Front Window Gland:
    addPart("unitCyl", 95.2, ly, 18.25, 0, 0, kin.angE * 2.0, 0.76, 0.76, 13.5, MATS.steel, 11);
    addPart("bevelGear", 95.2, ly, 22.80, 0, 0, kin.angE * 2.0, 0.46, 0.46, 0.46, MATS.brassHi, 11);
    addPart("bevelGear", 95.2, -22.45, 24.0, Math.PI * 0.5, 0, -kin.angE, 0.46, 0.46, 0.46, MATS.copper, 11);
    // Upper Miter Outrigger Bearing Boss & Ruby Chaton supporting the vertical arbor top at [95.2, ly, 24.65] and connecting to declBarrel's front window gland
    addPart("unitCyl", 95.2, ly, 24.65, 0, 0, 0, 1.32, 1.32, 0.70, MATS.brass, 11);
    addPart("jewelChaton", 95.2, ly, 25.05, 0, 0, 0, 0.72, 0.72, 0.65, MATS.brassHi, 11);
    addPart("oilCup", 95.2, ly, 25.25, 0, 0, 0, 0.58, 0.58, 0.58, MATS.brassHi, 11);
    addPart("unitCyl", 95.2, -21.5, 24.65, Math.PI * 0.5, 0, 0, 0.68, 0.68, 4.2, MATS.brass, 11);
    // Short front bevel stub axle [95.2, -22.50..-19.6, 24.0] seated in front window bearing gland + Twin-Side Yoke Coupling around plunger slot to rear crank disk!
    addPart("unitCyl", 95.2, -21.05, 24.0, Math.PI * 0.5, 0, -kin.angE, 0.72, 0.72, 3.0, MATS.steel, 11);
    addPart("unitCyl", 95.2, -19.8, 24.0, Math.PI * 0.5, 0, 0, 1.42, 1.42, 1.2, MATS.brassLo, 11);
    addPart("jupMoonRing", 95.2, -11.8, 24.0, Math.PI * 0.5, 0, -kin.angE, 0.86, 0.86, 0.75, MATS.brassHi, 11);
    for (const sgnX of [-1, 1]) {
      addPart("unitCyl", 95.2 + sgnX * 4.6, -13.8, 24.0, Math.PI * 0.5, 0, 0, 0.34, 0.34, 9.2, MATS.steel, 11);
    }
    // Branch B: Pitch-tangent intermediate idler at [69.70, -13.215, 14.55] (rPitch = 5.833, 5.25 + 5.833 = 11.083 exact!) driving Kidney Cam base gear
    addPart("gear_gal_ca", 69.70, -13.215, 14.55, 0, 0, kin.angE * (5.25 / 5.833), 1.3888, 1.3888, 0.50, MATS.copper, 11);
    addPart("unitCyl", 69.70, -13.215, 13.55, 0, 0, 0, 1.12, 1.12, 1.1, MATS.brassLo, 11);
    addPart("unitCyl", 69.70, -13.215, 14.95, 0, 0, kin.angE * (5.25 / 5.833), 0.72, 0.72, 1.7, MATS.steel, 11);
    addPart("oilCup", 69.70, -13.215, 15.88, 0, 0, 0, 0.65, 0.65, 0.65, MATS.brassHi, 11);
    // Branch C: Pitch-tangent idler at [69.25, 6.22, 14.55] (rPitch = 5.116) coupling Kidney Cam (5.25+5.116=10.366) to Geneva Driver (5.116+3.08=8.196)
    addPart("gear_gal_ca", 69.25, 6.22, 14.55, 0, 0, kin.angE * (5.25 / 5.116), 1.218, 1.218, 0.50, MATS.brass, 11);
    addPart("unitCyl", 69.25, 6.22, 13.55, 0, 0, 0, 0.98, 0.98, 1.1, MATS.brassLo, 11);
    addPart("unitCyl", 69.25, 6.22, 14.85, 0, 0, kin.angE * (5.25 / 5.116), 0.72, 0.72, 1.5, MATS.steel, 11);
    addPart("oilCup", 69.25, 6.22, 15.68, 0, 0, 0, 0.65, 0.65, 0.65, MATS.brassHi, 11);

    const [gx, gy] = DECL_XY;
    addPart("unitCyl", gx, gy, 14.4, 0, 0, 0, 9.6, 9.6, 2.8, MATS.brass, 11);
    addPart("declBarrel", gx, gy, 15.8, 0, 0, 0, 1, 1, 1, MATS.brass, 11, 1);
    addPart("unitCyl", gx, gy, 33.2, 0, 0, 0, 8.8, 8.8, 2.8, MATS.brass, 11);
    addPart("unitCyl", gx, gy, 36.2, 0, 0, 0, 4.4, 4.4, 3.2, MATS.brassLo, 11);
    // 4 External Tie-Rod Columns & Butterfly Wing-Nuts clamping declBarrel
    for (const tieAng of [0.4, 1.3, 1.85, 2.75]) {
      const tx = gx + Math.cos(tieAng) * 9.1;
      const ty = gy + Math.sin(tieAng) * 9.1;
      addPart("unitCyl", tx, ty, 24.8, 0, 0, 0, 0.44, 0.44, 19.2, MATS.steel, 11);
      addPart("daVinciWingNut", tx, ty, 34.4, 0, 0, tieAng, 0.62, 0.62, 0.62, MATS.brassHi, 11);
    }

    // Internal Scotch-Yoke Crank Disk (supported on a rear coaxial journal in declBarrel's back wall) + True Circular Crank Pin (rCrank = 2.85 in both X and Z!)
    const strokeAmp = 2.85;
    const pinX = gx - Math.cos(kin.angE) * strokeAmp;
    const pinZ = 24.0 + Math.sin(kin.angE) * strokeAmp;
    addPart("unitCyl", gx, gy + 5.2, 24.0, Math.PI * 0.5, 0, -kin.angE, 0.78, 0.78, 5.6, MATS.steel, 11);
    addPart("unitCyl", gx, gy + 2.0, 24.0, Math.PI * 0.5, 0, -kin.angE, 4.4, 4.4, 1.4, MATS.copper, 11);
    addPart("unitCyl", pinX, gy, pinZ, Math.PI * 0.5, 0, 0, 0.82, 0.82, 3.8, MATS.ruby, 11);

    const plungerBaseZ = 28.0 + Math.sin(kin.angE) * strokeAmp;
    addPart("scotchYokePlunger", gx, gy, plungerBaseZ, 0, 0, 0, 1, 1, 1, MATS.chrome, 11);

    // Graduated +-23.44° Scale Bar bolted directly onto declBarrel's -X top flange (z = 34.6, r = 6.85 < 8.8!) with flush +X graduation ticks facing the pointer
    addPart("scaleBar", gx - 6.8, gy - 0.8, 34.6, 0, 0, 0, 1, 1, 1, MATS.brassHi, 11);
    for (let t = -4; t <= 4; t++) {
      const tz = 40.9 + t * 0.7125;
      const major = t % 2 === 0;
      addPart("unitCyl", gx - 5.45, gy - 0.8, tz, 0, Math.PI * 0.5, 0, 0.18, 0.18, major ? 0.42 : 0.28, MATS.bluedSteel, 11);
    }

    // Equation-of-Time Reniform Kidney Cam & Rigid-Length Bell-Crank Sector Lever at CAM_XY = [74.0, -3.0]
    const [cxCam, cyCam] = CAM_XY;
    const rFollowerRoller = 0.95;
    addPart("unitCyl", cxCam, cyCam, 13.55, 0, 0, 0, 3.2, 3.2, 1.1, MATS.brassLo, 11);
    addPart("gear_gal_ca", cxCam, cyCam, 14.55, 0, 0, -kin.angE, 1.25, 1.25, 0.50, MATS.brassHi, 11);
    addPart("eotKidneyCam", cxCam, cyCam, 16.5, 0, 0, -kin.angE, 1, 1, 1, MATS.copper, 11);
    addPart("unitCyl", cxCam, cyCam, 16.4, 0, 0, -kin.angE, 0.84, 0.84, 4.8, MATS.bluedSteel, 11);
    // Pierced High-Arch Swiss Bridge Cock, Sunken Jewel Chaton, Slotted Screw & Oil Cup over the Kidney Cam Arbor
    // Oriented at rz = 0.32 rad so its foot anchors onto a solid deck pillar at [66.15, -5.60] clear of all gears and Saros disc!
    const camCockAng = 0.32;
    const camFootX = cxCam - Math.cos(camCockAng) * 10.6 * 0.78;
    const camFootY = cyCam - Math.sin(camCockAng) * 10.6 * 0.78;
    addPart("horologicalCock", cxCam, cyCam, 14.8, 0, 0, camCockAng, 0.78, 0.78, 0.90, MATS.brassHi, 11);
    addPart("unitCyl", camFootX, camFootY, 14.5, 0, 0, 0, 0.95, 0.95, 3.0, MATS.brassLo, 11);
    addPart("jewelChaton", cxCam, cyCam, 18.85, 0, 0, 0, 0.92, 0.92, 0.85, MATS.brassHi, 11);
    addPart("slottedScrew", camFootX, camFootY, 16.00, 0, 0, 0.3, 0.72, 0.72, 0.85, MATS.bluedSteel, 11);
    addPart("oilCup", cxCam, cyCam, 19.08, 0, 0, 0, 0.68, 0.68, 0.68, MATS.brassHi, 11);

    // Rigid Bell-Crank Sector Lever (L = 13.000 exact) + Coaxial Blued-Steel Hairspring at pivot [79.8, 10.0, 16.8]:
    const eotPivotX = 79.8, eotPivotY = 10.0;
    const eotCockAng = -Math.PI * 0.45;
    const eotFootX = eotPivotX - Math.cos(eotCockAng) * 10.6 * 0.76;
    const eotFootY = eotPivotY - Math.sin(eotCockAng) * 10.6 * 0.76;
    addPart("unitCyl", eotPivotX, eotPivotY, 16.1, 0, 0, 0, 0.92, 0.92, 6.2, MATS.brassHi, 11);
    addPart("hairspring", eotPivotX, eotPivotY, 15.9, 0, 0, kin.eotLeverAngle * 0.5, 0.82, 0.82, 0.85, MATS.bluedSteel, 11);
    addPart("eotSectorLever", eotPivotX, eotPivotY, 16.8, 0, 0, kin.eotLeverAngle, 1, 1, 1, MATS.brassHi, 11);
    addPart("horologicalCock", eotPivotX, eotPivotY, 14.8, 0, 0, eotCockAng, 0.76, 0.76, 0.90, MATS.brassHi, 11);
    addPart("unitCyl", eotFootX, eotFootY, 14.5, 0, 0, 0, 0.92, 0.92, 3.0, MATS.brassLo, 11);
    addPart("jewelChaton", eotPivotX, eotPivotY, 18.85, 0, 0, 0, 0.88, 0.88, 0.85, MATS.brassHi, 11);
    addPart("slottedScrew", eotFootX, eotFootY, 16.00, 0, 0, -0.4, 0.70, 0.70, 0.85, MATS.bluedSteel, 11);
    addPart("oilCup", eotPivotX, eotPivotY, 19.08, 0, 0, 0, 0.65, 0.65, 0.65, MATS.brassHi, 11);
    addPart("unitCyl", kin.eotRollerX, kin.eotRollerY, 16.8, 0, 0, 0, rFollowerRoller, rFollowerRoller, 2.2, MATS.ruby, 11);
    // Sector differential output pinion at [89.39, 16.53, 16.8] (distance 9.50 + 2.10 = 11.60 along +34.25° sector bisector!)
    // Phase-locked at exact pitch ratio 9.50 / 2.10 so sector teeth mesh cleanly into pinion root valleys without interpenetration!
    // + Lower compound step-up wheel at [89.39, 16.53, 15.6] (rPitch = 4.20 * 0.8955 = 3.761) meshing at exact pitch tangency (3.761 + 3.893 = 7.654) with Saros idler [96.70, 14.26, 15.6]!
    const secMeshAng = Math.atan2(16.53 - 10.0, 89.39 - 79.8);
    const secStartLocal = ((120 - 19.895) * Math.PI) / 180;
    const secPinionRot =
      secMeshAng +
      Math.PI -
      ((kin.eotLeverAngle - secMeshAng + secStartLocal) * (9.50 / 2.10) + Math.PI / 12);
    addPart("unitCyl", 89.39, 16.53, 13.9, 0, 0, 0, 1.15, 1.15, 1.8, MATS.brassLo, 11);
    addPart("gear_gal_io", 89.39, 16.53, 16.8, 0, 0, secPinionRot, 1.0, 1.0, 0.65, MATS.copper, 11);
    addPart("gear_gal_ca", 89.39, 16.53, 15.6, 0, 0, secPinionRot, 0.8955, 0.8955, 0.65, MATS.brass, 11);
    addPart("unitCyl", 89.39, 16.53, 15.25, 0, 0, secPinionRot, 0.58, 0.58, 4.5, MATS.bluedSteel, 11);
    addPart("oilCup", 89.39, 16.53, 17.50, 0, 0, 0, 0.62, 0.62, 0.62, MATS.brassHi, 11);

    addPickSphere(gx + 2, gy, 34.0, 15.0, 11);
    addPickSphere(cxCam, cyCam, 17.0, 10.0, 11);
  }

  // ========================================================================
  // GROUP 12: Antikythera Spiral Saros, 4-Hand Astronomical Dial & 6-Slot Maltese Cross Geneva Drive
  // ========================================================================
  {
    // 6-Slot Maltese Cross (Geneva) Intermittent Annual Indexing Drive at [65.5, 13.5] -> [65.5, 20.7] (d_G = 7.20 exact):
    addPart("unitCyl", 65.5, 13.5, 13.55, 0, 0, 0, 2.2, 2.2, 1.1, MATS.brassLo, 12);
    addPart("unitCyl", 65.5, 20.7, 13.55, 0, 0, 0, 2.2, 2.2, 1.1, MATS.brassLo, 12);
    addPart("unitCyl", 65.5, 13.5, 15.8, 0, 0, -kin.angE * (5.25 / 3.08), 0.68, 0.68, 5.6, MATS.steel, 12);
    addPart("unitCyl", 65.5, 20.7, 15.8, 0, 0, kin.genevaAngle, 0.72, 0.72, 5.6, MATS.steel, 12);
    addPart("gear_gal_eu", 65.5, 13.5, 14.55, 0, 0, -kin.angE * (5.25 / 3.08), 1.1, 1.1, 0.50, MATS.brassHi, 12);
    addPart("genevaDriver", 65.5, 13.5, 15.6, 0, 0, Math.PI * 0.5 - kin.genevaDriverAngle, 1, 1, 0.62, MATS.copper, 12);
    addPart("genevaWheel", 65.5, 20.7, 15.6, 0, 0, kin.genevaAngle, 1, 1, 0.62, MATS.brassHi, 12);
    addPart("hand_eot", 65.5, 20.7, 16.4, 0, 0, kin.genevaAngle, 0.34, 0.34, 0.7, MATS.bluedSteel, 12);
    addPart("unitCyl", 65.5, 20.7, 16.7, 0, 0, 0, 0.75, 0.75, 0.6, MATS.ruby, 12);
    // 2-Jewel Swiss Geneva Bridge ("Pont de Croix de Malte") spanning [65.5, 13.5] -> [65.5, 20.7] + Sautoir Detent Jumper Spring:
    addPart("swissTrainBridge", 65.5, 13.5, 17.5, 0, 0, Math.PI * 0.5, 0.30, 0.30, 0.45, MATS.brassHi, 12);
    // Stepped brass bearing bushes filling the [65.5, 13.5] and [65.5, 20.7] bridge ring bores + twin outrigger pillars at [65.5 +- 6.66, 17.28]
    addPart("unitCyl", 65.5, 13.5, 17.65, 0, 0, 0, 2.25, 2.25, 0.95, MATS.brassHi, 12);
    addPart("unitCyl", 65.5, 20.7, 17.65, 0, 0, 0, 1.45, 1.45, 0.95, MATS.brassHi, 12);
    addPart("unitCyl", 58.84, 17.28, 15.55, 0, 0, 0, 0.72, 0.72, 5.1, MATS.brassLo, 12);
    addPart("unitCyl", 72.16, 17.28, 15.55, 0, 0, 0, 0.72, 0.72, 5.1, MATS.brassLo, 12);
    addPart("jewelChaton", 65.5, 13.5, 18.25, 0, 0, 0, 0.82, 0.82, 0.75, MATS.brassHi, 12);
    addPart("jewelChaton", 65.5, 20.7, 18.25, 0, 0, 0, 0.82, 0.82, 0.75, MATS.brassHi, 12);
    addPart("oilCup", 65.5, 13.5, 18.48, 0, 0, 0, 0.58, 0.58, 0.58, MATS.brassHi, 12);
    addPart("oilCup", 65.5, 20.7, 18.48, 0, 0, 0, 0.58, 0.58, 0.58, MATS.brassHi, 12);
    addPart("slottedScrew", 58.84, 17.28, 18.10, 0, 0, 0.2, 0.55, 0.55, 0.75, MATS.bluedSteel, 12);
    addPart("slottedScrew", 72.16, 17.28, 18.10, 0, 0, -0.2, 0.55, 0.55, 0.75, MATS.bluedSteel, 12);
    const sautoirOsc = kin.genevaDwell ? 0.0 : -0.045 * Math.cos(kin.genevaDriverAngle * 3.0);
    // Anchor jumperSpring on solid ephemBox deck post at [75.0, 26.8, 15.6] (sx=0.84 -> tip length 7.056 at angle -2.0973 rad)
    // so V-beak lands in the 0° (+X) peripheral notch [71.45, 20.7] (r = 5.95 on genevaWheel) strictly inside ephemBox [57, 131]!
    addPart("unitCyl", 75.0, 26.8, 14.3, 0, 0, 0, 1.15, 1.15, 2.6, MATS.brassLo, 12);
    addPart("jumperSpring", 75.0, 26.8, 15.6, 0, 0, -2.0973 + sautoirOsc, 0.84, 0.84, 0.95, MATS.bluedSteel, 12);
    addPart("slottedScrew", 75.0, 26.8, 16.00, 0, 0, 0.5, 0.65, 0.65, 0.85, MATS.bluedSteel, 12);

    // Coplanar spur coupling idler at [96.70, 14.26, 15.6] linking Sector Pinion [89.39, 16.53] to Saros Column [104.0, 12.0]
    const secMeshAng12 = Math.atan2(16.53 - 10.0, 89.39 - 79.8);
    const secStartLocal12 = ((120 - 19.895) * Math.PI) / 180;
    const secPinionRot12 =
      secMeshAng12 +
      Math.PI -
      ((kin.eotLeverAngle - secMeshAng12 + secStartLocal12) * (9.50 / 2.10) + Math.PI / 12);
    addPart("gear_gal_ca", 96.70, 14.26, 15.6, 0, 0, -secPinionRot12 * (3.761 / 3.893), 0.927, 0.927, 0.65, MATS.brassHi, 12);
    addPart("unitCyl", 96.70, 14.26, 14.0, 0, 0, 0, 0.82, 0.82, 2.0, MATS.brassLo, 12);
    addPart("unitCyl", 96.70, 14.26, 15.6, 0, 0, -secPinionRot12 * (3.761 / 3.893), 0.64, 0.64, 1.3, MATS.steel, 12);
    addPart("oilCup", 96.70, 14.26, 16.30, 0, 0, 0, 0.62, 0.62, 0.62, MATS.brassHi, 12);

    const [dx, dy] = DIAL_XY;
    const dialZ = 45.5 + exp * 12.0;
    const eotAngle = kin.angE + (kin.eotMin / 60) * 1.8;

    // Driven EoT differential base gear on Saros Dial Column at [104.0, 12.0, 15.6] meshing at exact pitch tangency (3.893 + 3.748 = 7.641) with differential idler!
    addPart("gear_gal_ca", dx, dy, 15.6, 0, 0, secPinionRot12 * (3.761 / 3.748), 0.8924, 0.8924, 0.65, MATS.copper, 12);
    addPart("unitCyl", dx, dy, 14.0, 0, 0, 0, 5.4, 5.4, 2.0, MATS.brass, 12);
    addPart("unitCyl", dx, dy, 15.6, 0, 0, secPinionRot12 * (3.761 / 3.748), 0.58, 0.58, 1.2, MATS.steel, 12);

    // 3 Concentric Telescoping Coaxial Hand Sleeves + Under-Dial Équation Marchante Epicyclic Differential & 19:223 Draconic Reduction Train (z = 16.2..dialZ):
    // 1) Innermost Mean Solar & Saros Arbor (r = 1.04, rotating at kin.angE)
    const hMeanArbor = (dialZ + 7.4) - 16.2;
    addPart("unitCyl", dx, dy, 16.2 + hMeanArbor * 0.5, 0, 0, kin.angE, 1.04, 1.04, hMeanArbor, MATS.steel, 12);
    // 2) Middle True-Sun Équation Marchante Coaxial Sleeve (r = 1.48, rIn = 1.15 > 1.04, rotating at eotAngle)
    const hEotSleeve = (dialZ + 6.9) - 23.0;
    addPart("unitTube", dx, dy, 23.0 + hEotSleeve * 0.5, 0, 0, eotAngle, 1.48, 1.48, hEotSleeve, MATS.copper, 12);
    // 3) Outer Draconic Eclipse Node Coaxial Sleeve (r = 2.02, rIn = 1.57 > 1.48, rotating at kin.angNodal)
    const hNodeSleeve = (dialZ + 6.5) - 31.0;
    addPart("unitTube", dx, dy, 31.0 + hNodeSleeve * 0.5, 0, 0, kin.angNodal, 2.02, 2.02, hNodeSleeve, MATS.brass, 12);

    // Under-Dial Équation Marchante Planetary Differential Cage (z = 21.8..28.5) summing Mean Sun (ω_E) + Cam Equation (Δθ_EoT):
    // Planetary bevel gears at r = 1.80 have their local +Z cone apexes pointing INWARD ([-pCs, -pSn, 0]) toward [dx, dy, 26.2],
    // carried on a physical horizontal cross-axle inside a compact annular carrier ring (r = 9.2 * 0.285 = 2.62, clearing [dx - 5.3, dy]!)
    addPart("gear_lunarHub", dx, dy, 21.8, 0, 0, kin.angE, 0.75, 0.75, 0.9, MATS.brassHi, 12);
    addPart("bevelGear", dx, dy, 25.15, 0, 0, kin.angE, 0.56, 0.56, 0.52, MATS.brassHi, 12);
    addPart("jupMoonRing", dx, dy, 26.2, 0, 0, eotAngle, 0.285, 0.285, 0.85, MATS.copper, 12);
    const pDiffCs = Math.cos(eotAngle), pDiffSn = Math.sin(eotAngle);
    addSegmentRod(
      [dx - pDiffCs * 2.62, dy - pDiffSn * 2.62, 26.2],
      [dx + pDiffCs * 2.62, dy + pDiffSn * 2.62, 26.2],
      0.28,
      MATS.steel,
      12
    );
    for (const sgnP of [-1, 1]) {
      const pAng = eotAngle + (sgnP > 0 ? 0 : Math.PI);
      const pCs = Math.cos(pAng), pSn = Math.sin(pAng);
      addPart(
        "bevelGear",
        dx + pCs * 1.80,
        dy + pSn * 1.80,
        26.2,
        Math.atan2(pSn, 0),
        Math.asin(Math.max(-1, Math.min(1, -pCs))),
        -kin.angE * 2.0,
        0.34,
        0.34,
        0.34,
        MATS.brassHi,
        12
      );
      addPart("unitCyl", dx + pCs * 2.62, dy + pSn * 2.62, 26.2, Math.atan2(-pSn, 0), Math.asin(Math.max(-1, Math.min(1, pCs))), 0, 0.46, 0.46, 0.42, MATS.brassHi, 12);
    }
    addPart("bevelGear", dx, dy, 27.25, Math.PI, 0, eotAngle, 0.56, 0.56, 0.52, MATS.copper, 12);

    // Mid-Column Draconic Step-Down Idler Arbor [dx - 5.3, dy, z = 19.5..35.4] (elevated above z = 15.6 sarosIdler!)
    // with coaxial upper horologicalCock eye at [dx - 5.3, dy, 34.6..35.4] and full-height turned balusterPillar at [92.128, 12.0]:
    const sarosCockAng = 0.0;
    const sarosPillarX = (dx - 5.3) - 10.6 * 0.62;
    addSegmentRod([dx, dy, 19.5], [dx - 5.3, dy, 19.5], 0.42, MATS.brassLo, 12);
    addPart("unitCyl", dx - 5.3, dy, 19.5, 0, 0, 0, 0.92, 0.92, 0.9, MATS.brassLo, 12);
    addPart("unitCyl", dx - 5.3, dy, 23.0, 0, 0, -kin.angE * (3.30 / 2.00), 0.58, 0.58, 7.0, MATS.bluedSteel, 12);
    addPart("gear_gal_io", dx - 5.3, dy, 21.8, 0, 0, -kin.angE * (3.30 / 2.00), 0.9524, 0.9524, 0.70, MATS.copper, 12);
    addSegmentRod([sarosPillarX, dy, 27.0], [dx - 5.3, dy, 27.0], 0.36, MATS.brassLo, 12);
    addPart("unitCyl", dx - 5.3, dy, 27.0, 0, 0, 0, 0.82, 0.82, 1.0, MATS.brassLo, 12);
    addPart("unitCyl", dx - 5.3, dy, 31.45, 0, 0, -kin.angNodal * (4.224 / 1.076), 0.38, 0.38, 7.9, MATS.bluedSteel, 12);
    addPart("gear_jup_ga", dx - 5.3, dy, 32.2, 0, 0, -kin.angNodal * (4.224 / 1.076), 1.076 / 3.50, 1.076 / 3.50, 0.55, MATS.brassHi, 12);
    addPart("gear_lunarHub", dx, dy, 32.2, 0, 0, kin.angNodal, 0.96, 0.96, 0.55, MATS.copper, 12);
    addPart("balusterPillar", sarosPillarX, dy, 22.4, 0, 0, 0, 0.85, 0.85, 2.35, MATS.brassLo, 12);
    addPart("unitCyl", sarosPillarX, dy, 22.8, 0, 0, 0, 0.62, 0.62, 19.6, MATS.steel, 12);
    addPart("horologicalCock", dx - 5.3, dy, 31.8, 0, 0, sarosCockAng, 0.62, 0.72, 0.75, MATS.brassHi, 12);
    addPart("slottedScrew", sarosPillarX, dy, 32.80, 0, 0, 0.4, 0.65, 0.65, 0.80, MATS.bluedSteel, 12);
    addPart("jewelChaton", dx - 5.3, dy, 35.42, 0, 0, 0, 0.68, 0.68, 0.65, MATS.brassHi, 12);
    addPart("oilCup", dx - 5.3, dy, 35.62, 0, 0, 0, 0.55, 0.55, 0.55, MATS.brassHi, 12);

    // Cutaway-Enabled Saros Drum (cutawayFlag = 1) & Internal Antikythera e3/e4/f1/f2 (223-Month) + Exeligmos (h1/h2/i1) Epicyclic Gear Train:
    addPart("unitCyl", dx, dy, dialZ + 1.8, 0, 0, 0, 21.0, 21.0, 3.6, MATS.greekBronze, 12, 1);
    addPart("dialBezel", dx, dy, dialZ + 2.8, 0, 0, 0, 1, 1, 1, MATS.brassHi, 12, 1);
    // Internal Saros & Exeligmos gear train inside drum (z = dialZ + 4.10, revealed when X-Ray Cutaway [C] is toggled!)
    const sarosIdlerScale = (Math.hypot(8.4, 3.6) - 6.38) / 3.50;
    const subDialGearScale = (Math.hypot(8.0, 8.0) - 6.38) / 2.80;
    addPart("gear_lunarHub", dx, dy, dialZ + 4.1, 0, 0, kin.angE, 1.45, 1.45, 0.36, MATS.brassHi, 12);
    addPart("gear_gal_ga", dx - 8.4, dy + 3.6, dialZ + 4.1, 0, 0, -kin.angE * (6.38 / (3.50 * sarosIdlerScale)), sarosIdlerScale, sarosIdlerScale, 0.45, MATS.copper, 12);
    addPart("unitCyl", dx - 8.4, dy + 3.6, dialZ + 4.35, 0, 0, -kin.angE * (6.38 / (3.50 * sarosIdlerScale)), 0.42, 0.42, 1.3, MATS.bluedSteel, 12);
    // Exeligmos (54-yr = 3 Saros) & Callippic (76-yr = 4 Metonic) subsidiary arbors at [dx +- 8.0, dy - 8.0]
    const exelX = dx - 8.0, exelY = dy - 8.0;
    const callX = dx + 8.0, callY = dy - 8.0;
    addPart("gear_gal_eu", exelX, exelY, dialZ + 4.1, 0, 0, -kin.angE * (6.38 / (2.80 * subDialGearScale)), subDialGearScale, subDialGearScale, 0.45, MATS.copper, 12);
    addPart("unitCyl", exelX, exelY, dialZ + 4.90, 0, 0, -kin.angE * (6.38 / (2.80 * subDialGearScale)), 0.24, 0.24, 2.4, MATS.steel, 12);
    addPart("gear_gal_eu", callX, callY, dialZ + 4.1, 0, 0, -kin.angE * (6.38 / (2.80 * subDialGearScale)), subDialGearScale, subDialGearScale, 0.45, MATS.brassHi, 12);
    addPart("unitCyl", callX, callY, dialZ + 4.90, 0, 0, -kin.angE * (6.38 / (2.80 * subDialGearScale)), 0.24, 0.24, 2.4, MATS.steel, 12);

    // dialFace placed at dialZ + 5.40 so its top surface is at dialZ + 6.20, leaving sarosSpiralTrack and sub-dial bezels/hands 100% unburied above the dial plate!
    addPart("dialFace", dx, dy, dialZ + 5.40, 0, 0, 0, 1, 1, 1, dialFaceMat, 12);
    // 3D Machined 4-Turn Archimedes Spiral Guide Rail on Saros Dial Face (r = 4.6..8.2) + Raised Exeligmos (0/+8h/+16h) & Callippic Sub-Dial Registers
    addPart("sarosSpiralTrack", dx, dy, dialZ + 6.20, 0, 0, 0, 1, 1, 1, MATS.brassHi, 12);
    for (const [subX, subY, subAng, subMat] of [
      [exelX, exelY, (kin.angE / 3.0) % (Math.PI * 2), MATS.bluedSteel],
      [callX, callY, (-kin.angE / 4.0) % (Math.PI * 2), MATS.copper],
    ]) {
      addPart("unitTube", subX, subY, dialZ + 6.28, 0, 0, 0, 2.55, 2.55, 0.24, MATS.brassHi, 12);
      addPart("hand_eot", subX, subY, dialZ + 6.38, 0, 0, subAng, 0.15, 0.24, 0.36, subMat, 12);
      addPart("unitCyl", subX, subY, dialZ + 6.46, 0, 0, 0, 0.42, 0.42, 0.16, MATS.brassHi, 12);
      addPart("unitCyl", subX, subY, dialZ + 6.55, 0, 0, 0, 0.32, 0.32, 0.18, MATS.ruby, 12);
    }

    // 1) Draconic Eclipse Node Hand (Dragon Head/Tail, Blued Steel)
    addPart("hand_dragon", dx, dy, dialZ + 6.62, 0, 0, kin.angNodal, 1, 1, 1, MATS.bluedSteel, 12);
    // 2) Equation-of-Time True Sun Hand (Copper)
    addPart("hand_eot", dx, dy, dialZ + 7.02, 0, 0, eotAngle, 1, 1, 1, MATS.copper, 12);
    // 3) Breguet Mean Solar Year Hand (Gilded Brass) + Uniform-Circular Coaxial Slotted Radial Saros Guide Arm (sx = sy = 1.60!)
    addPart("hand_sun", dx, dy, dialZ + 7.36, 0, 0, kin.angE, 1, 1, 1, MATS.brassHi, 12);
    addPart("antikytheraSlotArm", dx, dy, dialZ + 7.54, 0, 0, kin.angE, 1.60, 1.60, 0.35, MATS.brassHi, 12);
    // 4) Antikythera Spiral Rider Shoe & Ruby Pin sliding radially along the 4-turn Saros spiral (r = 4.6 .. 8.2, strictly inside sub-dial ring!)
    const sarosTurn = ((kin.day / T_EARTH) % 4 + 4) % 4;
    const riderR = 4.6 + (sarosTurn / 4.0) * 3.6;
    const rx = dx + Math.cos(kin.angE) * riderR;
    const ry = dy + Math.sin(kin.angE) * riderR;
    addPart("transferBox", rx, ry, dialZ + 7.26, 0, 0, kin.angE, 0.095, 0.085, 0.045, MATS.copper, 12);
    addPart("unitCyl", rx, ry, dialZ + 7.62, 0, 0, 0, 0.72, 0.72, 1.4, MATS.ruby, 12);
    addPart("jewelChaton", rx, ry, dialZ + 7.86, 0, 0, kin.angE, 0.65, 0.65, 0.55, MATS.brassHi, 12);

    // Center Ruby Jewel Chaton & Cap
    addPart("jewelChaton", dx, dy, dialZ + 7.75, 0, 0, 0, 1.45, 1.45, 0.78, MATS.brassHi, 12);
    addPart("unitCyl", dx, dy, dialZ + 7.86, 0, 0, 0, 1.42, 1.42, 0.65, MATS.ruby, 12);

    // Front-Right ephemBox Well [115.5, -12.5, z = 13.0..20.8]: Antikythera 19-Year Metonic (235-Month) & 4-Year Olympiad Sub-Turret
    // Driven via 3-tier pitch-tangent companion countershaft at [122.0, -12.5] (D = 6.500 exact, clear of declBarrel!)
    const metX = 115.5, metY = -12.5;
    const metCx = 122.0, metCy = -12.5;
    addPart("unitCyl", metCx, metCy, 13.80, 0, 0, 0, 1.12, 1.12, 1.6, MATS.brassLo, 12);
    addPart("unitCyl", metCx, metCy, 16.30, 0, 0, -kin.angE * (3.50 / 3.00), 0.55, 0.55, 4.2, MATS.steel, 12);
    addPart("gear_gal_eu", metCx, metCy, 15.2, 0, 0, -kin.angE * (3.50 / 3.00), 3.00 / 2.80, 3.00 / 2.80, 0.55, MATS.brassHi, 12);
    addPart("unitTube", metCx, metCy, 16.5, 0, 0, -(kin.angE / 4.0) * (3.80 / 2.70), 0.70, 0.70, 0.75, MATS.brassLo, 12);
    addPart("gear_gal_io", metCx, metCy, 16.5, 0, 0, -(kin.angE / 4.0) * (3.80 / 2.70), 2.70 / 2.10, 2.70 / 2.10, 0.55, MATS.copper, 12);
    addPart("unitTube", metCx, metCy, 17.7, 0, 0, -(kin.angE / 19.0) * (4.20 / 2.30), 0.70, 0.70, 0.75, MATS.brassLo, 12);
    addPart("gear_gal_io", metCx, metCy, 17.7, 0, 0, -(kin.angE / 19.0) * (4.20 / 2.30), 2.30 / 2.10, 2.30 / 2.10, 0.55, MATS.brassHi, 12);
    addPart("oilCup", metCx, metCy, 18.35, 0, 0, 0, 0.58, 0.58, 0.58, MATS.brassHi, 12);

    addPart("unitCyl", metX, metY, 13.80, 0, 0, 0, 3.2, 3.2, 1.6, MATS.brassLo, 12);
    addPart("unitCyl", metX, metY, 16.8, 0, 0, kin.angE / 19.0, 0.64, 0.64, 7.4, MATS.steel, 12);
    addPart("unitTube", metX, metY, 15.2, 0, 0, kin.angE, 0.82, 0.82, 0.80, MATS.brassLo, 12);
    addPart("gear_jup_ga", metX, metY, 15.2, 0, 0, kin.angE, 1.0, 1.0, 0.55, MATS.copper, 12);
    addPart("unitTube", metX, metY, 16.5, 0, 0, kin.angE / 4.0, 0.82, 0.82, 0.80, MATS.brassLo, 12);
    addPart("gear_jup_eu", metX, metY, 16.5, 0, 0, kin.angE / 4.0, 3.80 / 2.80, 3.80 / 2.80, 0.55, MATS.brassHi, 12);
    addPart("gear_jup_io", metX, metY, 17.7, 0, 0, kin.angE / 19.0, 4.20 / 2.10, 4.20 / 2.10, 0.55, MATS.copper, 12);
    // High-arch Swiss bridge cock over Metonic/Olympiad sub-turret anchored at [115.5, -20.56] inside ephemBox
    const metCockAng = Math.PI * 0.5;
    const metFootY = metY - 10.6 * 0.76;
    addPart("horologicalCock", metX, metY, 15.2, 0, 0, metCockAng, 0.76, 0.76, 0.92, MATS.brassHi, 12);
    addPart("unitCyl", metX, metFootY, 14.7, 0, 0, 0, 0.92, 0.92, 3.4, MATS.brassLo, 12);
    addPart("slottedScrew", metX, metFootY, 16.40, 0, 0, 0.2, 0.68, 0.68, 0.85, MATS.bluedSteel, 12);
    // Coaxial Engraved 5-Turn Metonic & 4-Sector Olympiad Sub-Dial on Top of Bridge Cock (z = 19.2..20.8)
    addPart("dialBezel", metX, metY, 19.2, 0, 0, 0, 0.32, 0.32, 0.26, MATS.brassHi, 12);
    addPart("dialFace", metX, metY, 19.72, 0, 0, 0, 0.32, 0.32, 0.42, dialFaceMat, 12);
    addPart("jewelChaton", metX, metY, 20.10, 0, 0, 0, 0.42, 0.42, 0.42, MATS.brassHi, 12);
    addPart("hand_eot", metX, metY, 20.32, 0, 0, kin.angE / 4.0, 0.28, 0.28, 0.40, MATS.copper, 12);
    addPart("hand_sun", metX, metY, 20.54, 0, 0, kin.angE / 19.0, 0.31, 0.31, 0.42, MATS.bluedSteel, 12);
    addPart("unitCyl", metX, metY, 20.66, 0, 0, 0, 0.50, 0.50, 0.16, MATS.brassHi, 12);
    addPart("unitCyl", metX, metY, 20.75, 0, 0, 0, 0.36, 0.36, 0.20, MATS.ruby, 12);

    addPickSphere(dx, dy, dialZ + 5.0, 21.0, 12);
    addPickSphere(65.5, 17.5, 16.0, 9.0, 12);
    addPickSphere(metX, metY, 18.0, 9.0, 12);
  }

  return { earthPos, moonPos };
}

export function inspectOrrerySceneAtDay(day, geometry, lightTheme = false, cutawayMode = 0, explodeMode = 0) {
  const kin = evaluateKinematics(day);
  const parts = [];
  const pickSpheres = [];
  let overlayVertCount = 0;
  const scratchTRS = new Float32Array(32);

  function addPart(meshName, tx, ty, tz, rx, ry, rz, sx, sy, sz, mat, groupId, cutawayFlag = 0) {
    const mesh = geometry.meshes[meshName];
    if (!mesh) return;
    composeTRS(scratchTRS, tx, ty, tz, rx, ry, rz, sx, sy, sz);
    parts.push({
      meshName,
      tx, ty, tz,
      rx, ry, rz,
      sx, sy, sz,
      mat,
      groupId,
      cutawayFlag,
      matrix: new Float32Array(scratchTRS.subarray(0, 16)),
      normalMatrix: new Float32Array(scratchTRS.subarray(16, 32)),
    });
  }

  function addSegmentRod(p0, p1, radius, mat, groupId, isTube = false) {
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 0.001) return;
    const mx = (p0[0] + p1[0]) * 0.5;
    const my = (p0[1] + p1[1]) * 0.5;
    const mz = (p0[2] + p1[2]) * 0.5;
    const ux = dx / len, uy = dy / len, uz = dz / len;
    const ry = Math.asin(Math.max(-1, Math.min(1, ux)));
    const rx = Math.atan2(-uy, uz);
    const prevLen = parts.length;
    addPart(isTube ? "unitTube" : "unitCyl", mx, my, mz, rx, ry, 0, radius, radius, len, mat, groupId);
    if (parts.length > prevLen) {
      parts[parts.length - 1].rodEndpoints = [p0, p1];
    }
  }

  function addPickSphere(x, y, z, r, groupId) {
    pickSpheres.push({ x, y, z, r, groupId });
  }

  function pushOverlayVert() {
    overlayVertCount++;
  }

  const { earthPos, moonPos } = populateOrreryScene(
    geometry,
    addPart,
    addSegmentRod,
    addPickSphere,
    pushOverlayVert,
    kin,
    [0, -180, 120],
    lightTheme,
    cutawayMode,
    explodeMode
  );

  return { kin, parts, pickSpheres, overlayVertCount, earthPos, moonPos };
}

export async function initOrreryWebGPU({ canvas, shaderSource, geometry, isLightTheme, onTelemetryChange }) {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter available.");
  }
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (ev) => {
    console.error("WebGPU uncaptured error:", ev.error ? ev.error.message : ev);
  });
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: "premultiplied",
  });

  // 1. Static Vertex & Index Buffers for CAD Meshes
  const vertexBuffer = device.createBuffer({
    size: geometry.vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(vertexBuffer.getMappedRange()).set(geometry.vertexData);
  vertexBuffer.unmap();

  const indexBuffer = device.createBuffer({
    size: geometry.indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(indexBuffer.getMappedRange()).set(geometry.indexData);
  indexBuffer.unmap();

  // 2. Pre-Allocated Instance Storage Buffer (max 1280 rigid sub-parts * 40 floats = 204,800 bytes)
  const MAX_INSTANCES = 1280;
  const FLOATS_PER_INST = 40; // 160 bytes
  const instanceData = new Float32Array(MAX_INSTANCES * FLOATS_PER_INST);
  const instanceBuffer = device.createBuffer({
    size: instanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // 3. Pre-Allocated Scene Uniform Buffer (64 floats = 256 bytes)
  const sceneUniformData = new Float32Array(64);
  const sceneUniformBuffer = device.createBuffer({
    size: sceneUniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 4. Pre-Allocated Dynamic Overlay Vertex Buffer (24576 vertices)
  const MAX_OVERLAY_VERTS = 24576;
  const OVERLAY_FLOATS_PER_VERT = 12; // 48 bytes
  const overlayData = new Float32Array(MAX_OVERLAY_VERTS * OVERLAY_FLOATS_PER_VERT);
  const overlayBuffer = device.createBuffer({
    size: overlayData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // 5. Shadow Map (2048x2048 depth32float) & Comparison Sampler
  const SHADOW_SIZE = 2048;
  const shadowTexture = device.createTexture({
    size: [SHADOW_SIZE, SHADOW_SIZE, 1],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const shadowView = shadowTexture.createView();
  const shadowSampler = device.createSampler({
    compare: "less-equal",
    magFilter: "linear",
    minFilter: "linear",
  });

  // 6. Explicit BindGroupLayouts
  const shadowBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });
  const shadowPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [shadowBindGroupLayout],
  });
  const shadowBindGroup = device.createBindGroup({
    layout: shadowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
      { binding: 1, resource: { buffer: instanceBuffer } },
    ],
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup0 = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: sceneUniformBuffer } },
      { binding: 1, resource: { buffer: instanceBuffer } },
      { binding: 2, resource: shadowView },
      { binding: 3, resource: shadowSampler },
    ],
  });

  // 7. Compile WGSL Shader Module & Pipelines
  const shaderModule = device.createShaderModule({ code: shaderSource });

  const cadVertexBufferLayout = {
    arrayStride: 36,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x2" },
      { shaderLocation: 3, offset: 32, format: "float32" },
    ],
  };

  const shadowPipeline = device.createRenderPipeline({
    layout: shadowPipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_shadow",
      buffers: [cadVertexBufferLayout],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const SAMPLE_COUNT = 4;
  const mainPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [cadVertexBufferLayout],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: true,
      depthCompare: "less-equal",
    },
    multisample: { count: SAMPLE_COUNT },
  });

  const overlayPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_overlay",
      buffers: [
        {
          arrayStride: 48,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x4" },
            { shaderLocation: 3, offset: 40, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_overlay",
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: false,
      depthCompare: "less-equal",
    },
    multisample: { count: SAMPLE_COUNT },
  });

  // 8. MSAA 4x Color & Depth Attachment Management
  let msaaColorTex = null;
  let msaaColorView = null;
  let depthTex = null;
  let depthView = null;
  let curWidth = 0;
  let curHeight = 0;

  function ensureAttachments(w, h) {
    if (w === curWidth && h === curHeight && msaaColorView && depthView) return;
    curWidth = w;
    curHeight = h;
    if (msaaColorTex) msaaColorTex.destroy();
    if (depthTex) depthTex.destroy();

    msaaColorTex = device.createTexture({
      size: [w, h, 1],
      sampleCount: SAMPLE_COUNT,
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    msaaColorView = msaaColorTex.createView();

    depthTex = device.createTexture({
      size: [w, h, 1],
      sampleCount: SAMPLE_COUNT,
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTex.createView();
  }

  // Pre-allocated Draw Calls & Picking Spheres
  const drawCalls = [];
  for (let i = 0; i < MAX_INSTANCES; i++) {
    drawCalls.push({ mesh: null, instIndex: i });
  }
  let activeDraws = 0;

  const pickSpheres = [];
  for (let i = 0; i < 96; i++) {
    pickSpheres.push({ x: 0, y: 0, z: 0, r: 1, groupId: 0 });
  }
  let activePickSpheres = 0;

  function addPickSphere(x, y, z, r, groupId) {
    if (activePickSpheres >= pickSpheres.length) return;
    const s = pickSpheres[activePickSpheres++];
    s.x = x; s.y = y; s.z = z; s.r = r; s.groupId = groupId;
  }

  const scratchTRS = new Float32Array(32);

  function addPart(meshName, tx, ty, tz, rx, ry, rz, sx, sy, sz, mat, groupId, cutawayFlag = 0) {
    const mesh = geometry.meshes[meshName];
    if (!mesh || activeDraws >= MAX_INSTANCES) return;
    const idx = activeDraws++;
    drawCalls[idx].mesh = mesh;
    drawCalls[idx].instIndex = idx;

    const base = idx * FLOATS_PER_INST;
    composeTRS(scratchTRS, tx, ty, tz, rx, ry, rz, sx, sy, sz);
    instanceData.set(scratchTRS, base);

    instanceData[base + 32] = mat.rgb[0];
    instanceData[base + 33] = mat.rgb[1];
    instanceData[base + 34] = mat.rgb[2];
    instanceData[base + 35] = mat.id;

    instanceData[base + 36] = groupId;
    instanceData[base + 37] = mat.rough;
    instanceData[base + 38] = mat.metal;
    instanceData[base + 39] = cutawayFlag;
  }

  function addSegmentRod(p0, p1, radius, mat, groupId, isTube = false) {
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 0.001) return;
    const mx = (p0[0] + p1[0]) * 0.5;
    const my = (p0[1] + p1[1]) * 0.5;
    const mz = (p0[2] + p1[2]) * 0.5;
    const ux = dx / len, uy = dy / len, uz = dz / len;
    const ry = Math.asin(Math.max(-1, Math.min(1, ux)));
    const rx = Math.atan2(-uy, uz);
    addPart(isTube ? "unitTube" : "unitCyl", mx, my, mz, rx, ry, 0, radius, radius, len, mat, groupId);
  }

  let overlayVertCount = 0;
  function pushOverlayVert(x, y, z, nx, ny, nz, r, g, b, a, u, v) {
    if (overlayVertCount >= MAX_OVERLAY_VERTS) return;
    const base = overlayVertCount * OVERLAY_FLOATS_PER_VERT;
    overlayData[base + 0] = x;
    overlayData[base + 1] = y;
    overlayData[base + 2] = z;
    overlayData[base + 3] = nx;
    overlayData[base + 4] = ny;
    overlayData[base + 5] = nz;
    overlayData[base + 6] = r;
    overlayData[base + 7] = g;
    overlayData[base + 8] = b;
    overlayData[base + 9] = a;
    overlayData[base + 10] = u;
    overlayData[base + 11] = v;
    overlayVertCount++;
  }

  const CENTER_XY = [-8.0, 0.0];
  const ECLIPTIC_PLANE_Z = 109.0;
  const SUN_POS = [CENTER_XY[0], CENTER_XY[1], ECLIPTIC_PLANE_Z];
  const SUN_R = 6.8;

  let latestEarthPos = [-8 + 45, 0, ECLIPTIC_PLANE_Z];
  let latestMoonPos = [-8 + 54, 0, ECLIPTIC_PLANE_Z];

  function buildSceneAtDay(kin, eyePos, lightTheme, cutawayMode, explodeMode) {
    activeDraws = 0;
    activePickSpheres = 0;
    overlayVertCount = 0;
    const res = populateOrreryScene(
      geometry,
      addPart,
      addSegmentRod,
      addPickSphere,
      pushOverlayVert,
      kin,
      eyePos,
      lightTheme,
      cutawayMode,
      explodeMode
    );
    latestEarthPos = res.earthPos;
    latestMoonPos = res.moonPos;
  }

  // Camera & Interaction State
  const state = {
    preset: "overview",
    az: CAMERA_PRESETS.overview.az,
    el: CAMERA_PRESETS.overview.el,
    dist: CAMERA_PRESETS.overview.dist,
    target: [...CAMERA_PRESETS.overview.target],
    fov: CAMERA_PRESETS.overview.fov,

    goalAz: CAMERA_PRESETS.overview.az,
    goalEl: CAMERA_PRESETS.overview.el,
    goalDist: CAMERA_PRESETS.overview.dist,
    goalTarget: [...CAMERA_PRESETS.overview.target],
    goalFov: CAMERA_PRESETS.overview.fov,

    day: 45,
    goalDay: 45,
    playing: true,
    speedDaysPerSec: 18,
    cutaway: 0,
    goalCutaway: 0,
    explode: 0,
    goalExplode: 0,
    hoveredGroup: 0,
    selectedGroup: 0,
    isLightTheme: Boolean(isLightTheme),
    lastFrameMs: 0,
    avgFrameMs: 8.0,
  };

  const matView = new Float32Array(16);
  const matProj = new Float32Array(16);
  const matViewProj = new Float32Array(16);
  const matInvViewProj = new Float32Array(16);
  const matLightView = new Float32Array(16);
  const matLightProj = new Float32Array(16);
  const matLightVP = new Float32Array(16);

  function pickAtNormalized(nx, ny) {
    const ndcX = nx * 2.0 - 1.0;
    const ndcY = 1.0 - ny * 2.0;

    function unproject(zNdc) {
      const x = matInvViewProj[0] * ndcX + matInvViewProj[4] * ndcY + matInvViewProj[8] * zNdc + matInvViewProj[12];
      const y = matInvViewProj[1] * ndcX + matInvViewProj[5] * ndcY + matInvViewProj[9] * zNdc + matInvViewProj[13];
      const z = matInvViewProj[2] * ndcX + matInvViewProj[6] * ndcY + matInvViewProj[10] * zNdc + matInvViewProj[14];
      const w = matInvViewProj[3] * ndcX + matInvViewProj[7] * ndcY + matInvViewProj[11] * zNdc + matInvViewProj[15] || 1;
      return [x / w, y / w, z / w];
    }

    const pNear = unproject(0.0);
    const pFar = unproject(1.0);
    let dx = pFar[0] - pNear[0], dy = pFar[1] - pNear[1], dz = pFar[2] - pNear[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len; dy /= len; dz /= len;

    let bestGroup = 0;
    let bestT = Infinity;
    for (let i = 0; i < activePickSpheres; i++) {
      const s = pickSpheres[i];
      const ox = pNear[0] - s.x, oy = pNear[1] - s.y, oz = pNear[2] - s.z;
      const b = ox * dx + oy * dy + oz * dz;
      const c = ox * ox + oy * oy + oz * oz - s.r * s.r;
      const disc = b * b - c;
      if (disc >= 0) {
        const t = -b - Math.sqrt(disc);
        if (t > 0 && t < bestT) {
          bestT = t;
          bestGroup = s.groupId;
        }
      }
    }
    return bestGroup;
  }

  function renderFrame(nowMs, dtSec) {
    if (state.playing && dtSec > 0) {
      state.goalDay += state.speedDaysPerSec * dtSec;
    }
    const alpha = 1 - Math.exp(-Math.max(0.001, dtSec) * 11.0);
    state.day += (state.goalDay - state.day) * alpha;
    state.cutaway += (state.goalCutaway - state.cutaway) * alpha;
    state.explode += (state.goalExplode - state.explode) * alpha;

    const kin = evaluateKinematics(state.day);

    if (state.preset === "tellurian") {
      const ex = CENTER_XY[0] + Math.cos(kin.angE) * 45.0;
      const ey = CENTER_XY[1] + Math.sin(kin.angE) * 45.0;
      state.goalTarget[0] = ex;
      state.goalTarget[1] = ey;
      state.goalTarget[2] = ECLIPTIC_PLANE_Z - 4.0;
    }

    state.az += (state.goalAz - state.az) * alpha;
    state.el += (state.goalEl - state.el) * alpha;
    state.dist += (state.goalDist - state.dist) * alpha;
    state.fov += (state.goalFov - state.fov) * alpha;
    for (let k = 0; k < 3; k++) {
      state.target[k] += (state.goalTarget[k] - state.target[k]) * alpha;
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(64, Math.round(rect.width * dpr));
    const h = Math.max(64, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ensureAttachments(w, h);

    const azRad = (state.az * Math.PI) / 180;
    const elRad = (Math.max(6, Math.min(86, state.el)) * Math.PI) / 180;
    const eye = [
      state.target[0] + Math.cos(elRad) * Math.cos(azRad) * state.dist,
      state.target[1] + Math.cos(elRad) * Math.sin(azRad) * state.dist,
      state.target[2] + Math.sin(elRad) * state.dist,
    ];
    mat4LookAt(matView, eye, state.target, [0, 0, 1]);
    mat4PerspectiveZO(matProj, (state.fov * Math.PI) / 180, w / h, 8.0, 980.0);
    mat4Multiply(matViewProj, matProj, matView);
    mat4Invert(matInvViewProj, matViewProj);

    // Studio key directional light from front-left-top (-X, -Y, +Z) for crisp instrument relief
    const keyDir = [-0.36, -0.44, 0.82];
    const kLen = Math.hypot(keyDir[0], keyDir[1], keyDir[2]);
    keyDir[0] /= kLen; keyDir[1] /= kLen; keyDir[2] /= kLen;
    const lightEye = [keyDir[0] * 240, keyDir[1] * 240, 35 + keyDir[2] * 240];
    mat4LookAt(matLightView, lightEye, [0, 0, 35], [0, 0, 1]);
    mat4OrthoZO(matLightProj, -175, 175, -145, 145, 20, 500);
    mat4Multiply(matLightVP, matLightProj, matLightView);

    buildSceneAtDay(kin, eye, state.isLightTheme, state.cutaway, state.explode);

    sceneUniformData.set(matViewProj, 0);
    sceneUniformData.set(matLightVP, 16);
    sceneUniformData[32] = eye[0];
    sceneUniformData[33] = eye[1];
    sceneUniformData[34] = eye[2];
    sceneUniformData[35] = nowMs * 0.001;

    sceneUniformData[36] = SUN_POS[0];
    sceneUniformData[37] = SUN_POS[1];
    sceneUniformData[38] = SUN_POS[2] + state.explode * 40.0;
    sceneUniformData[39] = SUN_R;

    sceneUniformData[40] = latestEarthPos[0];
    sceneUniformData[41] = latestEarthPos[1];
    sceneUniformData[42] = latestEarthPos[2];
    sceneUniformData[43] = 4.2;

    sceneUniformData[44] = latestMoonPos[0];
    sceneUniformData[45] = latestMoonPos[1];
    sceneUniformData[46] = latestMoonPos[2];
    sceneUniformData[47] = 1.75;

    sceneUniformData[48] = keyDir[0];
    sceneUniformData[49] = keyDir[1];
    sceneUniformData[50] = keyDir[2];
    sceneUniformData[51] = state.isLightTheme ? 1.0 : 0.0;

    sceneUniformData[52] = state.hoveredGroup;
    sceneUniformData[53] = state.selectedGroup;
    sceneUniformData[54] = state.cutaway;
    sceneUniformData[55] = kin.eclipseStrength;

    sceneUniformData[56] = w;
    sceneUniformData[57] = h;
    sceneUniformData[58] = 1 / w;
    sceneUniformData[59] = 1 / h;

    device.queue.writeBuffer(sceneUniformBuffer, 0, sceneUniformData);
    device.queue.writeBuffer(instanceBuffer, 0, instanceData, 0, activeDraws * FLOATS_PER_INST);
    if (overlayVertCount > 0) {
      device.queue.writeBuffer(overlayBuffer, 0, overlayData, 0, overlayVertCount * OVERLAY_FLOATS_PER_VERT);
    }

    const encoder = device.createCommandEncoder();

    // Pass 1: Shadow Depth Map
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: shadowView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(shadowPipeline);
      pass.setBindGroup(0, shadowBindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, "uint32");
      for (let i = 0; i < activeDraws; i++) {
        const dc = drawCalls[i];
        pass.drawIndexed(dc.mesh.indexCount, 1, dc.mesh.firstIndex, 0, dc.instIndex);
      }
      pass.end();
    }

    // Pass 2 & Pass 3: Main MSAA Render Pass + Transparent Overlays resolved to Swapchain
    const swapView = context.getCurrentTexture().createView();
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: msaaColorView,
            resolveTarget: swapView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "discard",
          },
        ],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "discard",
        },
      });

      pass.setPipeline(mainPipeline);
      pass.setBindGroup(0, bindGroup0);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, "uint32");
      for (let i = 0; i < activeDraws; i++) {
        const dc = drawCalls[i];
        pass.drawIndexed(dc.mesh.indexCount, 1, dc.mesh.firstIndex, 0, dc.instIndex);
      }

      if (overlayVertCount > 0) {
        pass.setPipeline(overlayPipeline);
        pass.setBindGroup(0, bindGroup0);
        pass.setVertexBuffer(0, overlayBuffer);
        pass.draw(overlayVertCount, 1, 0, 0);
      }

      pass.end();
    }

    device.queue.submit([encoder.finish()]);

    if (onTelemetryChange) {
      onTelemetryChange(kin, state);
    }
    return kin;
  }

  function setPreset(name) {
    const p = CAMERA_PRESETS[name];
    if (!p) return;
    state.preset = name;
    state.goalAz = p.az;
    state.goalEl = p.el;
    state.goalDist = p.dist;
    state.goalFov = p.fov;
    if (Array.isArray(p.target)) {
      state.goalTarget[0] = p.target[0];
      state.goalTarget[1] = p.target[1];
      state.goalTarget[2] = p.target[2];
    }
  }

  function dispose() {
    try {
      context.unconfigure();
      device.destroy();
    } catch (_) {}
  }

  window.addEventListener("pagehide", (ev) => {
    if (!ev.persisted) dispose();
  });

  function projectWorldToViewBox(x, y, z, viewW = 620, viewH = 387.5) {
    const vp = matViewProj;
    const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (cw <= 0.01) return [-9999, -9999];
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    return [(ndcX * 0.5 + 0.5) * viewW, (0.5 - ndcY * 0.5) * viewH];
  }

  return {
    state,
    renderFrame,
    pickAtNormalized,
    projectWorldToViewBox,
    setPreset,
    dispose,
    getActiveDraws: () => activeDraws,
    getViewProj: () => matViewProj,
  };
}
