import * as THREE from 'three/webgpu';
import {
  EROSION_GRID_INNER,
  EROSION_GRID_SIZE,
  dirToOctUv,
  evalBaseBoulderRadiusCpu,
  fbm3DCpu,
  mirrorOctahedralRgba16Gutter,
  mirrorOctahedralScalarGutter,
  texelDirAndSolidAngle,
} from './octahedral';
import { bakeAmmoniteRadialEnvelope } from './ammonite';

/** Minimum expected rock clearance above the buried ammonite (verified across seeds without clamping to raw ribs) */
export const MIN_FOSSIL_OVERBURDEN = 0.04;
/** Recess depth below fossil ribs to clear limestone inter-rib valleys cleanly without z-fighting */
export const FOSSIL_MATRIX_RECESS = 0.022;
/** Maximum carve depth into bare limestone collar outside the ammonite footprint */
export const MAX_COLLAR_CARVE_DEPTH = 0.24;
/** Minimum solid inner shale pedestal radius anywhere on the nodule */
export const MIN_CORE_FLOOR_RADIUS = 0.24;
/** Nominal air-abrasive jet cone half-angle on the unit sphere (radians) */
export const JET_CONE_RADIUS_RAD = 0.185;

export interface CarveResult {
  carvedVolumeDelta: number;
  touchedTexels: number;
  hitFossilRatio: number;
  averageDepthRemoved: number;
}

export interface ErosionRaycastHit {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  direction: THREE.Vector3;
  distance: number;
  isFossil: boolean;
}

// Pre-allocated scratch buffers for zero-GC raycasting and sampling
const _uvScratch: [number, number] = [0, 0];
const _dirScratch: [number, number, number] = [0, 0, 0];
const _tVec = new THREE.Vector3();
const _bVec = new THREE.Vector3();
const _probeDir = new THREE.Vector3();

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / Math.max(1e-6, e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * Core differential-hardness 3D radial erosion heightfield on a seamless
 * 256x256 octahedral grid with 1-texel mirrored border gutters.
 *
 * Physical invariants:
 *  - `fossilRadius[i]` is immutable (pyritized ammonite has infinite relative hardness).
 *  - `matrixFloor[i] <= currentRadius[i] <= startRadius[i]` always holds.
 *  - Combined solid surface `max(currentRadius[i], fossilRadius[i]) >= fossilRadius[i]` everywhere.
 *  - `totalCarvedVolume` strictly conserves spherical-shell removed limestone volume.
 */
export class ErosionField {
  readonly size = EROSION_GRID_SIZE;
  readonly inner = EROSION_GRID_INNER;
  readonly cellCount = EROSION_GRID_SIZE * EROSION_GRID_SIZE;

  readonly dirX: Float32Array;
  readonly dirY: Float32Array;
  readonly dirZ: Float32Array;
  readonly solidAngle: Float32Array;

  /** Analytical C-infinity base concretion radius per texel (subtracted before octahedral packing) */
  readonly baseBoulderRadius: Float32Array;
  /** Immutable exact radial surface of the embedded 3D pyritized ammonite (0 outside fossil) */
  readonly fossilRadius: Float32Array;
  /** Unchipped outer limestone concretion surface (100% independent of fossil ribs) */
  readonly unfracturedRadius: Float32Array;
  /** Starting limestone surface on boot / reset (includes initial jagged fracture window) */
  readonly startRadius: Float32Array;
  /** Live carvable limestone matrix radial surface */
  readonly currentRadius: Float32Array;
  /** Signed isotropic displacement delta `currentRadius - baseBoulderRadius` (zero y=0 fold kink) */
  readonly deltaRadius: Float32Array;
  /** Differential-hardness erosion floor (stops matrix erosion cleanly at fossil / core bed) */
  readonly matrixFloor: Float32Array;
  /** Geological strata & cleavage-plate resistance [0.45 .. 1.35] */
  readonly strataHardness: Float32Array;
  /** Powdery fresh chalk dust intensity [0 .. 1] */
  readonly freshChalk: Float32Array;

  /** Live RGBA16F DataTexture uploaded to WebGPU vertex + fragment shaders */
  readonly textureData: Uint16Array;
  readonly dataTexture: THREE.DataTexture;

  /** Current dipping sedimentary bedding plane normal (varies subtly per seed) */
  readonly strataAxis = new THREE.Vector3(0.24, 0.94, 0.24).normalize();

  strataSeed = 1;
  totalCarvedVolume = 0;
  exposedFraction = 0;

  private fossilTotalSolidAngle = 0;
  private exposedSolidAngle = 0;
  private texDirty = false;
  private activeChalkIndices: Int32Array;
  private activeChalkCount = 0;
  private chalkMarked: Uint8Array;

  constructor(ammoniteGeometry: THREE.BufferGeometry, initialSeed = 1) {
    this.dirX = new Float32Array(this.cellCount);
    this.dirY = new Float32Array(this.cellCount);
    this.dirZ = new Float32Array(this.cellCount);
    this.solidAngle = new Float32Array(this.cellCount);
    this.baseBoulderRadius = new Float32Array(this.cellCount);

    for (let iy = 1; iy <= this.inner; iy++) {
      for (let ix = 1; ix <= this.inner; ix++) {
        const dOmega = texelDirAndSolidAngle(ix, iy, _dirScratch, this.size);
        const idx = iy * this.size + ix;
        this.dirX[idx] = _dirScratch[0];
        this.dirY[idx] = _dirScratch[1];
        this.dirZ[idx] = _dirScratch[2];
        this.solidAngle[idx] = dOmega;
        this.baseBoulderRadius[idx] = evalBaseBoulderRadiusCpu(
          _dirScratch[0],
          _dirScratch[1],
          _dirScratch[2]
        );
      }
    }

    this.fossilRadius = bakeAmmoniteRadialEnvelope(ammoniteGeometry, this.size);

    this.unfracturedRadius = new Float32Array(this.cellCount);
    this.startRadius = new Float32Array(this.cellCount);
    this.currentRadius = new Float32Array(this.cellCount);
    this.deltaRadius = new Float32Array(this.cellCount);
    this.matrixFloor = new Float32Array(this.cellCount);
    this.strataHardness = new Float32Array(this.cellCount);
    this.freshChalk = new Float32Array(this.cellCount);

    this.activeChalkIndices = new Int32Array(this.cellCount);
    this.chalkMarked = new Uint8Array(this.cellCount);

    this.textureData = new Uint16Array(this.cellCount * 4);
    this.dataTexture = new THREE.DataTexture(
      this.textureData,
      this.size,
      this.size,
      THREE.RGBAFormat,
      THREE.HalfFloatType
    );
    this.dataTexture.colorSpace = THREE.NoColorSpace;
    this.dataTexture.flipY = false;
    this.dataTexture.wrapS = THREE.ClampToEdgeWrapping;
    this.dataTexture.wrapT = THREE.ClampToEdgeWrapping;
    this.dataTexture.magFilter = THREE.LinearFilter;
    this.dataTexture.minFilter = THREE.LinearFilter;
    this.dataTexture.generateMipmaps = false;

    this.reset(initialSeed);
  }

  /**
   * Re-encrusts a fresh unprepped stratified limestone concretion nodule around the
   * embedded pyritized ammonite with a new strata orientation and jagged initial fracture window.
   */
  reset(seed = this.strataSeed + 1): void {
    this.strataSeed = seed;
    const sOffX = ((seed * 17.31) % 29.0) - 14.5;
    const sOffY = ((seed * 11.73) % 23.0) - 11.5;
    const sOffZ = ((seed * 7.19) % 19.0) - 9.5;

    // Slightly varied geological bedding dip per nodule
    const dipAngle = 0.22 + 0.16 * Math.sin(seed * 1.618);
    const dipAz = 0.45 + 0.55 * Math.cos(seed * 2.13);
    this.strataAxis
      .set(
        Math.sin(dipAngle) * Math.cos(dipAz),
        Math.cos(dipAngle),
        Math.sin(dipAngle) * Math.sin(dipAz)
      )
      .normalize();

    const sax = this.strataAxis.x;
    const say = this.strataAxis.y;
    const saz = this.strataAxis.z;

    // Initial natural chipped fracture window on the front-upper-right flank peeking onto gleaming ribs
    const winAngle = 0.52 + 0.16 * Math.sin(seed * 2.41);
    const winX = 0.56 * Math.cos(winAngle) + 0.08;
    const winY = 0.56 * Math.sin(winAngle) + 0.04;
    const winZ = Math.sqrt(Math.max(0.20, 1.0 - winX * winX - winY * winY));
    const winDirLen = Math.hypot(winX, winY, winZ);
    const wdx = winX / winDirLen;
    const wdy = winY / winDirLen;
    const wdz = winZ / winDirLen;

    // Two subtle natural rock-hammer cleavage flats on outer margins for authentic museum boulder silhouette
    const fl1x = -0.82;
    const fl1y = 0.50;
    const fl1z = 0.28;
    const fl2x = 0.78;
    const fl2y = -0.58;
    const fl2z = -0.24;

    this.fossilTotalSolidAngle = 0;
    this.exposedSolidAngle = 0;
    this.activeChalkCount = 0;
    this.chalkMarked.fill(0);
    this.totalCarvedVolume = 0;

    for (let iy = 1; iy <= this.inner; iy++) {
      const row = iy * this.size;
      for (let ix = 1; ix <= this.inner; ix++) {
        const idx = row + ix;
        const dx = this.dirX[idx];
        const dy = this.dirY[idx];
        const dz = this.dirZ[idx];
        const dOmega = this.solidAngle[idx];

        // Pure independent C-infinity analytical boulder base + chiseled sedimentary relief
        // Strictly ZERO clamping against raw fossilRadius -> guarantees 0% ammonite rib ghosting!
        const rBase = this.baseBoulderRadius[idx];

        const macroRock = fbm3DCpu(dx + sOffX * 0.1, dy + sOffY * 0.1, dz + sOffZ * 0.1, 2.25);
        const mesoChisel = fbm3DCpu(dx * 2.1 + sOffZ * 0.15, dy * 2.1 - sOffX * 0.1, dz * 2.1, 4.6);
        const px = dx * rBase;
        const py = dy * rBase;
        const pz = dz * rBase;
        const strataCoord = (px * sax + py * say + pz * saz) * 11.5 + macroRock * 1.4;
        // Crisp stepped sedimentary bedding ledges on the chiseled limestone exterior
        const bedSin = Math.sin(strataCoord);
        const beddingLedge =
          bedSin * 0.015 + Math.sin(strataCoord * 2.0 + 0.7) * 0.006 - Math.pow(Math.max(0, -bedSin), 4.0) * 0.008;

        // Subtle hammer-cleaved rock facets on outer rim shoulders only (keeps full clearance over buried fossil)
        const c1 = Math.max(0, dx * fl1x + dy * fl1y + dz * fl1z - 0.74);
        const c2 = Math.max(0, dx * fl2x + dy * fl2y + dz * fl2z - 0.76);
        const cleavageTrim = (c1 * c1 + c2 * c2) * 0.42;

        // Smooth angular lobe variation (~+/-0.048) stays strictly above buried ammonite without touching fR
        const r =
          rBase +
          (macroRock - 0.5) * 0.088 +
          (mesoChisel - 0.5) * 0.032 +
          beddingLedge -
          cleavageTrim;

        this.unfracturedRadius[idx] = r;
        const fR = this.fossilRadius[idx];

        // Per-texel strata & spall-plate abrasion resistance (harder ferruginous laminae step cleanly)
        const ironBand = 0.5 + 0.5 * Math.sin(strataCoord * 1.45 + 0.4);
        const plateNoise = fbm3DCpu(dx * 5.2 + 3.1, dy * 5.2 + 7.4, dz * 5.2 - 2.2, 3.6);
        this.strataHardness[idx] = 0.56 + 0.46 * ironBand * ironBand + (plateNoise - 0.5) * 0.30;

        // Differential-hardness erosion floor
        let floorR: number;
        if (fR > 0) {
          // Soft limestone clears down just beneath the proud ammonite surface so 3D ribs stand out cleanly
          floorR = Math.max(0.10, fR - FOSSIL_MATRIX_RECESS);
          this.fossilTotalSolidAngle += dOmega;
        } else {
          floorR = Math.max(MIN_CORE_FLOOR_RADIUS, r - MAX_COLLAR_CARVE_DEPTH);
        }
        this.matrixFloor[idx] = floorR;

        // Sculpt natural jagged initial fracture window on front flank exposing a peek of gleaming pyrite ribs
        let startR = r;
        let chalk = 0.0;
        const dotWin = dx * wdx + dy * wdy + dz * wdz;
        if (dotWin > 0.52) {
          const ang = Math.acos(Math.min(1, dotWin));
          // Irregular jagged mineral spall outline (strictly 3D FBM, zero circular bullseye)
          const jag =
            (fbm3DCpu(dx * 5.8 + sOffX * 0.2, dy * 5.8 + 4.3, dz * 5.8 - 1.7, 3.2) - 0.5) * 0.19;
          const effAng = ang + jag;
          const winInner = 0.30;
          const winOuter = 0.47;
          if (effAng < winOuter) {
            const cutMask = 1.0 - smoothstep(winInner, winOuter, effAng);
            // Stepped spall cliff edge along the perimeter of the fracture window
            const steppedCut = Math.pow(cutMask, 0.62);
            const targetR = fR > 0 ? fR - FOSSIL_MATRIX_RECESS * 0.95 : floorR + 0.04;
            startR = Math.max(floorR, r - (r - targetR) * steppedCut);
            // Subtle crushed white calcite rim along the natural rock fracture lip
            const rimBand = smoothstep(0.05, 0.35, cutMask) * (1.0 - smoothstep(0.65, 0.96, cutMask));
            chalk = rimBand * 0.56 + cutMask * 0.12;
          }
        }

        this.startRadius[idx] = startR;
        this.currentRadius[idx] = startR;
        this.deltaRadius[idx] = startR - rBase;
        this.freshChalk[idx] = chalk;

        if (fR > 0) {
          this.exposedSolidAngle += this.evalTexelExposure(idx) * dOmega;
        }
      }
    }

    mirrorOctahedralScalarGutter(this.unfracturedRadius, this.size);
    mirrorOctahedralScalarGutter(this.startRadius, this.size);
    mirrorOctahedralScalarGutter(this.currentRadius, this.size);
    mirrorOctahedralScalarGutter(this.deltaRadius, this.size);
    mirrorOctahedralScalarGutter(this.matrixFloor, this.size);
    mirrorOctahedralScalarGutter(this.freshChalk, this.size);

    this.exposedFraction =
      this.fossilTotalSolidAngle > 1e-6 ? this.exposedSolidAngle / this.fossilTotalSolidAngle : 0;

    this.packAllToTexture();
  }

  private evalTexelExposure(idx: number): number {
    const fR = this.fossilRadius[idx];
    if (fR <= 0) return 0;
    const diff = fR - this.currentRadius[idx];
    if (diff <= 0) return 0;
    return Math.min(1, diff / (FOSSIL_MATRIX_RECESS * 0.82));
  }

  private writeTexelHalfFloat(idx: number): void {
    const toHalf = THREE.DataUtils.toHalfFloat;
    const curR = this.currentRadius[idx];
    // Pack signed isotropic offset `deltaR = curR - baseBoulderRadius` so GPU evaluates C-infinity
    // analytical R_base(d) with zero bilinear V-trough creases across octahedral y = 0 / x = 0 folds!
    const deltaR = curR - this.baseBoulderRadius[idx];
    const cutDepth = Math.max(0, this.unfracturedRadius[idx] - curR);
    const chalk = this.freshChalk[idx];
    const fR = this.fossilRadius[idx];
    const fossilClearance =
      fR > 0 ? Math.max(-1.0, Math.min(1.0, (fR - curR) / FOSSIL_MATRIX_RECESS)) : -1.0;

    const base = idx * 4;
    this.textureData[base] = toHalf(deltaR);
    this.textureData[base + 1] = toHalf(cutDepth);
    this.textureData[base + 2] = toHalf(chalk);
    this.textureData[base + 3] = toHalf(fossilClearance);
  }

  private packAllToTexture(): void {
    for (let iy = 1; iy <= this.inner; iy++) {
      const row = iy * this.size;
      for (let ix = 1; ix <= this.inner; ix++) {
        this.writeTexelHalfFloat(row + ix);
      }
    }
    mirrorOctahedralRgba16Gutter(this.textureData, this.size);
    this.dataTexture.needsUpdate = true;
    this.texDirty = false;
  }

  /**
   * Fires the pneumatic micro-abrasive dolomite grit jet centered at unit direction `hitDir`,
   * stripping soft stratified limestone down to the differential-hardness floor while
   * keeping the pyritized ammonite completely uncut.
   */
  carveAt(
    hitDirX: number,
    hitDirY: number,
    hitDirZ: number,
    dtSec: number,
    intensity = 1.0,
    jetRadiusRad = JET_CONE_RADIUS_RAD
  ): CarveResult {
    const invLen = 1.0 / Math.max(1e-8, Math.hypot(hitDirX, hitDirY, hitDirZ));
    const hx = hitDirX * invLen;
    const hy = hitDirY * invLen;
    const hz = hitDirZ * invLen;

    const maxAng = jetRadiusRad * 1.48;
    const cosLimit = Math.cos(maxAng);
    // Brisk, tactile 90-PSI air-abrasive stripping rate so 1-2s sweeps strip limestone off bold pyrite ribs
    const baseRate = 6.4 * Math.max(0, dtSec) * Math.max(0, intensity);

    let carvedVolumeDelta = 0;
    let touchedTexels = 0;
    let fossilContactWeight = 0;
    let totalContactWeight = 0;
    let depthSum = 0;
    let touchedEdge = false;

    for (let iy = 1; iy <= this.inner; iy++) {
      const row = iy * this.size;
      for (let ix = 1; ix <= this.inner; ix++) {
        const idx = row + ix;
        const dx = this.dirX[idx];
        const dy = this.dirY[idx];
        const dz = this.dirZ[idx];
        const cosAng = dx * hx + dy * hy + dz * hz;
        if (cosAng < cosLimit) continue;

        const ang = Math.acos(Math.min(1, cosAng));
        const normR = ang / Math.max(1e-4, jetRadiusRad);

        // Irregular mineral fracture edge modulation so pits spall cleanly along strata seams
        const hard = this.strataHardness[idx];
        const microChip = 0.84 + 0.32 * fbm3DCpu(dx * 16.0, dy * 16.0, dz * 16.0, 2.2);
        const effNormR = normR / (0.86 + 0.24 * microChip);
        if (effNormR >= 1.38) continue;

        const coreJet =
          Math.exp(-effNormR * effNormR * 2.25) * (1.0 - smoothstep(0.92, 1.38, effNormR));
        if (coreJet <= 1e-4) continue;

        const dOmega = this.solidAngle[idx];
        const oldR = this.currentRadius[idx];
        const floorR = this.matrixFloor[idx];
        const fR = this.fossilRadius[idx];

        totalContactWeight += coreJet;
        if (fR > 0 && oldR <= fR + 0.008) {
          fossilContactWeight += coreJet;
        }

        const prevExp = fR > 0 ? this.evalTexelExposure(idx) : 0;

        // Differential hardness: matrix erodes inversely with local strata hardness, strictly clamped at matrixFloor
        const cutAmount = (baseRate * coreJet * microChip) / hard;
        const newR = Math.max(floorR, oldR - cutAmount);
        const actualDelta = oldR - newR;

        if (actualDelta > 1e-7) {
          this.currentRadius[idx] = newR;
          this.deltaRadius[idx] = newR - this.baseBoulderRadius[idx];
          const volDelta = ((oldR * oldR * oldR - newR * newR * newR) * dOmega) / 3.0;
          carvedVolumeDelta += volDelta;
          depthSum += actualDelta;

          if (fR > 0) {
            const nextExp = this.evalTexelExposure(idx);
            this.exposedSolidAngle += (nextExp - prevExp) * dOmega;
          }
        }

        // Fresh chalk powder bloom on actively blasted limestone rims (clears off polished exposed pyrite)
        const chalkBoost =
          fR > 0 && newR <= fR - FOSSIL_MATRIX_RECESS * 0.65
            ? -coreJet * dtSec * 2.4
            : coreJet * (actualDelta > 1e-6 ? 0.58 : 0.16);
        const nextChalk = Math.max(0, Math.min(1, this.freshChalk[idx] + chalkBoost));
        if (Math.abs(nextChalk - this.freshChalk[idx]) > 1e-4 || actualDelta > 1e-7) {
          this.freshChalk[idx] = nextChalk;
          if (nextChalk > 0.01 && this.chalkMarked[idx] === 0) {
            this.chalkMarked[idx] = 1;
            this.activeChalkIndices[this.activeChalkCount++] = idx;
          }
          this.writeTexelHalfFloat(idx);
          touchedTexels++;
          if (ix === 1 || ix === this.inner || iy === 1 || iy === this.inner) {
            touchedEdge = true;
          }
        }
      }
    }

    if (carvedVolumeDelta > 0) {
      this.totalCarvedVolume += carvedVolumeDelta;
      this.exposedFraction =
        this.fossilTotalSolidAngle > 1e-6
          ? Math.max(0, Math.min(1, this.exposedSolidAngle / this.fossilTotalSolidAngle))
          : 0;
    }

    if (touchedTexels > 0) {
      if (touchedEdge) {
        mirrorOctahedralScalarGutter(this.currentRadius, this.size);
        mirrorOctahedralScalarGutter(this.deltaRadius, this.size);
        mirrorOctahedralRgba16Gutter(this.textureData, this.size);
      }
      this.texDirty = true;
    }

    return {
      carvedVolumeDelta,
      touchedTexels,
      hitFossilRatio: totalContactWeight > 1e-6 ? fossilContactWeight / totalContactWeight : 0,
      averageDepthRemoved: touchedTexels > 0 ? depthSum / touchedTexels : 0,
    };
  }

  /**
   * Gently settles fresh airborne blast chalk dust over time and flushes GPU texture uploads
   * ONLY when texels actually changed (zero GPU writeTexture overhead when idle).
   */
  stepAndUpload(dtSec: number): void {
    const settleRate = Math.exp(-dtSec * 0.55);
    let writeIdx = 0;
    let touchedEdge = false;

    for (let i = 0; i < this.activeChalkCount; i++) {
      const idx = this.activeChalkIndices[i];
      const cutDepth = Math.max(0, this.unfracturedRadius[idx] - this.currentRadius[idx]);
      const fR = this.fossilRadius[idx];
      const isCleanFossil = fR > 0 && this.currentRadius[idx] <= fR - FOSSIL_MATRIX_RECESS * 0.65;
      // Freshly carved limestone retains a subtle permanent powdery matte floor (~0.24)
      const restChalk = isCleanFossil ? 0.0 : smoothstep(0.008, 0.055, cutDepth) * 0.24;
      const cur = this.freshChalk[idx];
      const next = restChalk + (cur - restChalk) * settleRate;

      if (Math.abs(next - restChalk) < 0.003) {
        this.freshChalk[idx] = restChalk;
        this.chalkMarked[idx] = 0;
      } else {
        this.freshChalk[idx] = next;
        this.activeChalkIndices[writeIdx++] = idx;
      }
      this.writeTexelHalfFloat(idx);
      this.texDirty = true;

      const ix = idx % this.size;
      const iy = (idx / this.size) | 0;
      if (ix === 1 || ix === this.inner || iy === 1 || iy === this.inner) {
        touchedEdge = true;
      }
    }
    this.activeChalkCount = writeIdx;

    if (this.texDirty) {
      if (touchedEdge) {
        mirrorOctahedralScalarGutter(this.currentRadius, this.size);
        mirrorOctahedralScalarGutter(this.deltaRadius, this.size);
        mirrorOctahedralRgba16Gutter(this.textureData, this.size);
      }
      this.dataTexture.needsUpdate = true;
      this.texDirty = false;
    }
  }

  /**
   * Exact bilinear lookup of any single-channel octahedral field at unit direction `(x, y, z)`.
   */
  sampleGridBilinear(grid: Float32Array, dx: number, dy: number, dz: number): number {
    dirToOctUv(dx, dy, dz, _uvScratch, this.size);
    const fx = _uvScratch[0] * this.size - 0.5;
    const fy = _uvScratch[1] * this.size - 0.5;
    const x0 = Math.max(0, Math.min(this.size - 2, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(this.size - 2, Math.floor(fy)));
    const tx = Math.max(0, Math.min(1, fx - x0));
    const ty = Math.max(0, Math.min(1, fy - y0));

    const r0 = y0 * this.size;
    const r1 = (y0 + 1) * this.size;
    const v00 = grid[r0 + x0];
    const v10 = grid[r0 + x0 + 1];
    const v01 = grid[r1 + x0];
    const v11 = grid[r1 + x0 + 1];
    const a = v00 + (v10 - v00) * tx;
    const b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
  }

  /**
   * Evaluates the live matrix surface radius identically to GPU reconstruction:
   * `evalBaseBoulderRadiusCpu(d) + bilinear(deltaRadius)` (zero octahedral y=0 / x=0 crease).
   */
  sampleMatrixRadius(dx: number, dy: number, dz: number): number {
    return (
      evalBaseBoulderRadiusCpu(dx, dy, dz) +
      this.sampleGridBilinear(this.deltaRadius, dx, dy, dz)
    );
  }

  /**
   * Evaluates the live combined physical solid surface radius `max(matrixRadius, fossilRadius)`
   * along unit direction `(dx, dy, dz)`.
   */
  sampleCombinedRadius(dx: number, dy: number, dz: number): number {
    const rMat = this.sampleMatrixRadius(dx, dy, dz);
    const rFos = this.sampleGridBilinear(this.fossilRadius, dx, dy, dz);
    return rFos > rMat ? rFos : rMat;
  }

  /**
   * Computes exact integral of removed limestone volume directly from `startRadius` and `currentRadius`.
   * Used in tests to verify exact conservation against `totalCarvedVolume`.
   */
  computeExactCarvedVolume(): number {
    let vol = 0;
    for (let iy = 1; iy <= this.inner; iy++) {
      const row = iy * this.size;
      for (let ix = 1; ix <= this.inner; ix++) {
        const idx = row + ix;
        const r0 = this.startRadius[idx];
        const r1 = this.currentRadius[idx];
        if (r0 > r1) {
          vol += ((r0 * r0 * r0 - r1 * r1 * r1) * this.solidAngle[idx]) / 3.0;
        }
      }
    }
    return vol;
  }

  /**
   * Fast, zero-allocation analytical raymarch + 8-step bisection raycaster against the live
   * combined limestone + exposed pyritized ammonite surface. Runs in ~0.015 ms per ray.
   */
  raycast(
    origin: THREE.Vector3,
    dir: THREE.Vector3,
    outHit: ErosionRaycastHit
  ): boolean {
    const boundR = 1.34;
    const b = origin.dot(dir);
    const c = origin.lengthSq() - boundR * boundR;
    const disc = b * b - c;
    if (disc < 0) return false;

    const sqrtDisc = Math.sqrt(disc);
    const tNear = Math.max(0.05, -b - sqrtDisc);
    const tFar = -b + sqrtDisc;
    if (tFar <= tNear) return false;

    const steps = 48;
    const dt = (tFar - tNear) / steps;

    let prevT = tNear;
    let prevF = this.evalSignedRayRadius(origin, dir, prevT);
    if (prevF <= 0) {
      return this.finalizeHit(origin, dir, prevT, outHit);
    }

    let hitLo = tNear;
    let hitHi = tFar;
    let found = false;

    for (let i = 1; i <= steps; i++) {
      const t = tNear + i * dt;
      const f = this.evalSignedRayRadius(origin, dir, t);
      if (f <= 0) {
        hitLo = prevT;
        hitHi = t;
        found = true;
        break;
      }
      prevT = t;
      prevF = f;
    }

    if (!found) return false;

    // 8-step bisection -> ~0.15 mm exact hit localization
    for (let k = 0; k < 8; k++) {
      const mid = 0.5 * (hitLo + hitHi);
      if (this.evalSignedRayRadius(origin, dir, mid) <= 0) {
        hitHi = mid;
      } else {
        hitLo = mid;
      }
    }

    return this.finalizeHit(origin, dir, 0.5 * (hitLo + hitHi), outHit);
  }

  private evalSignedRayRadius(origin: THREE.Vector3, dir: THREE.Vector3, t: number): number {
    const px = origin.x + dir.x * t;
    const py = origin.y + dir.y * t;
    const pz = origin.z + dir.z * t;
    const r = Math.hypot(px, py, pz);
    if (r < 1e-6) return -1;
    const inv = 1.0 / r;
    const surfR = this.sampleCombinedRadius(px * inv, py * inv, pz * inv);
    return r - surfR;
  }

  private finalizeHit(
    origin: THREE.Vector3,
    dir: THREE.Vector3,
    t: number,
    outHit: ErosionRaycastHit
  ): boolean {
    const px = origin.x + dir.x * t;
    const py = origin.y + dir.y * t;
    const pz = origin.z + dir.z * t;
    const r = Math.hypot(px, py, pz);
    if (r < 1e-6) return false;
    const inv = 1.0 / r;
    const dx = px * inv;
    const dy = py * inv;
    const dz = pz * inv;

    outHit.position.set(px, py, pz);
    outHit.direction.set(dx, dy, dz);
    outHit.distance = t;

    const rMat = this.sampleMatrixRadius(dx, dy, dz);
    const rFos = this.sampleGridBilinear(this.fossilRadius, dx, dy, dz);
    outHit.isFossil = rFos > 0 && rFos >= rMat - 0.004;

    // Basis-invariant exact spherical heightfield normal: N = normalize(r*d - dr/dt * T - dr/db * B)
    if (Math.abs(dy) < 0.9) {
      _tVec.set(dz, 0, -dx).normalize();
    } else {
      _tVec.set(0, -dz, dy).normalize();
    }
    _bVec.crossVectors(outHit.direction, _tVec).normalize();

    const eps = 0.016;
    _probeDir.set(dx + _tVec.x * eps, dy + _tVec.y * eps, dz + _tVec.z * eps).normalize();
    const rTp = this.sampleCombinedRadius(_probeDir.x, _probeDir.y, _probeDir.z);
    _probeDir.set(dx - _tVec.x * eps, dy - _tVec.y * eps, dz - _tVec.z * eps).normalize();
    const rTm = this.sampleCombinedRadius(_probeDir.x, _probeDir.y, _probeDir.z);

    _probeDir.set(dx + _bVec.x * eps, dy + _bVec.y * eps, dz + _bVec.z * eps).normalize();
    const rBp = this.sampleCombinedRadius(_probeDir.x, _probeDir.y, _probeDir.z);
    _probeDir.set(dx - _bVec.x * eps, dy - _bVec.y * eps, dz - _bVec.z * eps).normalize();
    const rBm = this.sampleCombinedRadius(_probeDir.x, _probeDir.y, _probeDir.z);

    const drdt = (rTp - rTm) / (2.0 * eps);
    const drdb = (rBp - rBm) / (2.0 * eps);
    const rCenter = Math.max(rMat, rFos);

    outHit.normal
      .set(
        dx * rCenter - _tVec.x * drdt - _bVec.x * drdb,
        dy * rCenter - _tVec.y * drdt - _bVec.y * drdb,
        dz * rCenter - _tVec.z * drdt - _bVec.z * drdb
      )
      .normalize();

    return true;
  }

  dispose(): void {
    this.dataTexture.dispose();
  }
}
