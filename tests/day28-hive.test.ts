import { describe, expect, it } from 'vitest';
import {
  CELL_INRADIUS,
  HEX_RADIUS,
  MagneticDipoleLattice,
  WALL_HALF_THICKNESS,
  buildHexLatticeLayout,
} from '../src/days/28-hive/magneticLattice';
import {
  BEZEL_INNER_CIRCUM_R,
  NEEDLE_HALF_LENGTH,
  buildBezelBoltGeometry,
  buildBrassInnerTrimGeometry,
  buildBronzeBezelGeometry,
  buildHoneycombMatrixGeometry,
  buildLimestoneSlabGeometry,
  buildLodestoneStylusGeometries,
  buildNeedleHalfGeometry,
  buildPivotPinGeometry,
  buildUndersideMedallionGeometry,
  computeSignedVolume,
} from '../src/days/28-hive/hiveGeometry';
import { createHiveMaterials } from '../src/days/28-hive/hiveMaterials';

describe('Day 28 - Hive: Coupled 91-Cell Magnetic Dipole Honeycomb Lattice', () => {
  const layout = buildHexLatticeLayout(HEX_RADIUS);

  it('1. Lattice Geometry, Clearances, Outward CCW Winding & TSL Material Construction', () => {
    // Radius-5 axial hex grid contains exactly 1 + 3*5*6 = 91 cells (6-fold symmetry)
    expect(layout.cellCount).toBe(91);

    // Every needle half-length must maintain >= 10% radial clearance inside the brass cell walls
    const innerCellInradius = CELL_INRADIUS - WALL_HALF_THICKNESS;
    expect(NEEDLE_HALF_LENGTH).toBeLessThan(innerCellInradius * 0.90);
    expect(innerCellInradius - NEEDLE_HALF_LENGTH).toBeGreaterThan(0.011);

    // Entire 91-cell honeycomb must sit cleanly inside the dark-bronze bezel perimeter
    const bezelInnerInradius = BEZEL_INNER_CIRCUM_R * (Math.sqrt(3) * 0.5);
    expect(layout.maxFlatExtent + WALL_HALF_THICKNESS).toBeLessThan(bezelInnerInradius - 0.04);

    // Verify exact neighbor reciprocity across all 91 cells
    let interiorCount = 0;
    for (const c of layout.cells) {
      let deg = 0;
      for (let d = 0; d < 6; d++) {
        const nbId = c.neighborByDir[d];
        if (nbId >= 0) {
          deg++;
          const opp = (d + 3) % 6;
          expect(layout.cells[nbId].neighborByDir[opp]).toBe(c.id);
        }
      }
      expect(deg).toBeGreaterThanOrEqual(3);
      expect(deg).toBeLessThanOrEqual(6);
      if (deg === 6) interiorCount++;
    }
    // R=5 has R=4 interior = 61 cells with 6 neighbors
    expect(interiorCount).toBe(61);

    // Every watertight closed solid geometry must have strictly positive signed volume (outward CCW)
    const limestoneGeo = buildLimestoneSlabGeometry();
    const bezelGeo = buildBronzeBezelGeometry();
    const trimGeo = buildBrassInnerTrimGeometry();
    const boltGeo = buildBezelBoltGeometry();
    const honeycombGeo = buildHoneycombMatrixGeometry(layout);
    const pivotGeo = buildPivotPinGeometry();
    const northGeo = buildNeedleHalfGeometry(1);
    const southGeo = buildNeedleHalfGeometry(-1);
    const underside = buildUndersideMedallionGeometry();
    const stylus = buildLodestoneStylusGeometries();

    expect(computeSignedVolume(limestoneGeo)).toBeGreaterThan(0.5);
    expect(computeSignedVolume(bezelGeo)).toBeGreaterThan(0.05);
    expect(computeSignedVolume(trimGeo)).toBeGreaterThan(0.005);
    expect(computeSignedVolume(boltGeo)).toBeGreaterThan(0.00005);
    expect(computeSignedVolume(honeycombGeo)).toBeGreaterThan(0.05);
    expect(computeSignedVolume(pivotGeo)).toBeGreaterThan(0.00001);
    expect(computeSignedVolume(northGeo)).toBeGreaterThan(0.00001);
    expect(computeSignedVolume(southGeo)).toBeGreaterThan(0.00001);
    expect(computeSignedVolume(underside.bronzePlate)).toBeGreaterThan(0.005);
    expect(computeSignedVolume(underside.brassRing)).toBeGreaterThan(0.001);
    expect(computeSignedVolume(stylus.lodestoneTip)).toBeGreaterThan(0.0001);
    expect(computeSignedVolume(stylus.polarityBand)).toBeGreaterThan(0.00005);
    expect(computeSignedVolume(stylus.brassBody)).toBeGreaterThan(0.0005);
    expect(computeSignedVolume(stylus.darkGrip)).toBeGreaterThan(0.0005);
    expect(computeSignedVolume(stylus.brassPommel)).toBeGreaterThan(0.0001);

    // North & South needle halves must have identical mirrored volume
    expect(computeSignedVolume(northGeo)).toBeCloseTo(computeSignedVolume(southGeo), 8);

    limestoneGeo.dispose();
    bezelGeo.dispose();
    trimGeo.dispose();
    boltGeo.dispose();
    honeycombGeo.dispose();
    pivotGeo.dispose();
    northGeo.dispose();
    southGeo.dispose();
    underside.bronzePlate.dispose();
    underside.brassRing.dispose();
    stylus.dispose();

    // TSL mineral/metal materials construct & dispose cleanly
    const mats = createHiveMaterials();
    expect(mats.limestone.colorNode).toBeDefined();
    expect(mats.limestone.normalNode).toBeDefined();
    expect(mats.darkBronze.colorNode).toBeDefined();
    expect(mats.satinBrass.colorNode).toBeDefined();
    expect(mats.needleNorthCinnabar.colorNode).toBeDefined();
    expect(mats.needleSouthPrussian.colorNode).toBeDefined();
    expect(mats.lodestone.colorNode).toBeDefined();
    expect(mats.polarityBand.colorNode).toBeDefined();
    expect(mats.darkGrip.colorNode).toBeDefined();
    mats.dispose();
  });

  it('2. Emergent Dipole-Dipole Coupling: local perturbation propagates spin waves to distant cells', () => {
    const sim = new MagneticDipoleLattice(layout, 202628);
    sim.stylus.active = false;

    // Settle completely into twin-vortex flux closure first
    for (let f = 0; f < 180; f++) {
      sim.step(1 / 60);
    }
    const baseTelem = sim.getTelemetry();
    // Twin-vortex flux closure exhibits non-trivial local coherence & low net ferromagnetic moment
    expect(baseTelem.vortexCount).toBeGreaterThanOrEqual(1);
    expect(baseTelem.settledCount).toBeGreaterThan(75);
    expect(baseTelem.dipoleEnergy).toBeLessThan(-50);

    // Snapshot distant right-flank needle angles before perturbing ONLY the far-left cell
    const farLeft = sim.findNearestCell(-1.05, 0.0, 0.3);
    const farRight = sim.findNearestCell(1.05, 0.0, 0.3);
    expect(farLeft).toBeGreaterThanOrEqual(0);
    expect(farRight).toBeGreaterThanOrEqual(0);

    const beforeMidAngles = new Float32Array(sim.angles);
    // Perturb ONLY the single far-left needle (zero external stylus, zero shock pulse)
    sim.angles[farLeft] += Math.PI * 0.85;
    sim.omegas[farLeft] = 18.0;
    sim.locked[farLeft] = 0;

    let maxMidCellMotion = 0;
    for (let f = 0; f < 60; f++) {
      sim.step(1 / 60);
      for (let i = 0; i < sim.count; i++) {
        if (i === farLeft) continue;
        const c = layout.cells[i];
        if (Math.hypot(c.x - layout.cells[farLeft].x, c.z - layout.cells[farLeft].z) > 0.35) {
          const delta = Math.abs(sim.angles[i] - beforeMidAngles[i]);
          if (delta > maxMidCellMotion) maxMidCellMotion = delta;
        }
      }
    }
    // Pure dipole-dipole coupling must cascade measurable motion to non-adjacent cells (> 0.35 units away)
    expect(maxMidCellMotion).toBeGreaterThan(0.05);
  });

  it('3. Stylus Polarity, Thermal Self-Organization & Coulomb Static-Friction Settling', () => {
    const sim = new MagneticDipoleLattice(layout, 202628);

    // A. Thermal agitation scatters all needles; real dipole coupling lowers energy & self-organizes
    sim.stylus.active = false;
    sim.triggerThermalScatter(1.0);
    for (let f = 0; f < 6; f++) sim.step(1 / 60);
    const hotTelem = sim.getTelemetry();
    expect(hotTelem.meanSpeed).toBeGreaterThan(3.5);

    // Step 4.5 s to verify passive relaxation lowers dipole energy dramatically,
    // increases emergent local neighbor coherence, and cools angular speed
    for (let f = 0; f < 270; f++) {
      sim.step(1 / 60);
    }
    const coolTelem = sim.getTelemetry();
    expect(coolTelem.dipoleEnergy).toBeLessThan(hotTelem.dipoleEnergy - 45);
    expect(coolTelem.localCoherence).toBeGreaterThan(hotTelem.localCoherence + 0.25);
    expect(coolTelem.meanSpeed).toBeLessThan(0.20);
    expect(coolTelem.settledCount).toBeGreaterThanOrEqual(45);

    // B. External stylus attracts North (Cinnabar) tips toward wand when polarity = +1
    //    and flips cleanly away when polarity = -1
    const probeCell = sim.findNearestCell(0.0, 0.0, 0.2);
    expect(probeCell).toBeGreaterThanOrEqual(0);

    sim.stylus.active = true;
    sim.stylus.x = 0.25;
    sim.stylus.z = 0.0;
    sim.stylus.height = 0.14;
    sim.stylus.strength = 2.8;
    sim.stylus.polarity = 1;
    sim.stylus.vx = 0;
    sim.stylus.vz = 0;

    for (let f = 0; f < 90; f++) sim.step(1 / 60);
    // Needle at origin should point roughly toward +X (cos(theta) > 0.65)
    expect(Math.cos(sim.angles[probeCell])).toBeGreaterThan(0.65);

    // Flip polarity -> needle at origin should reverse toward -X (cos(theta) < -0.65)
    expect(sim.flipStylusPolarity()).toBe(-1);
    for (let f = 0; f < 120; f++) sim.step(1 / 60);
    expect(Math.cos(sim.angles[probeCell])).toBeLessThan(-0.65);
  });
});
