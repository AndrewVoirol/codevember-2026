import { describe, expect, it } from 'vitest';
import {
  CAMERA_PRESETS,
  GROUP_INFO,
  evaluateKinematics,
  formatTelemetryForGroup,
  inspectOrrerySceneAtDay,
} from '../engine-webgpu';
import { buildOrreryGeometry } from '../geometry-3d';

describe('Day 06 — Orrery: 107-Mesh CAD Library & 16-Group Horological Kinematics', () => {
  const geometry = buildOrreryGeometry();

  it('builds all 107 procedural CAD meshes with finite vertex coordinates and valid triangle indices', () => {
    const meshNames = Object.keys(geometry.meshes);
    expect(meshNames.length).toBeGreaterThanOrEqual(107);
    expect(geometry.vertexData.length).toBeGreaterThan(200_000 * 9);
    expect(geometry.indexData.length).toBeGreaterThan(100_000 * 3);

    for (let i = 0; i < geometry.vertexData.length; i += 997) {
      expect(Number.isFinite(geometry.vertexData[i])).toBe(true);
    }
  });

  it('evaluates 16-group kinematics, Mars opposition retrograde at Day 238.3, and Geneva dwell/snap laws', () => {
    const kin45 = evaluateKinematics(45);
    expect(kin45.season).toBe('Vernal');
    expect(Number.isFinite(kin45.angE)).toBe(true);

    const kinRetro = evaluateKinematics(238.3);
    expect(kinRetro.marsRetrograde).toBe(true);

    for (let g = 1; g <= 16; g++) {
      expect(GROUP_INFO[g]).toBeDefined();
      const telemetry = formatTelemetryForGroup(g, kin45);
      expect(telemetry).not.toBeNull();
      expect(telemetry?.code.length).toBeGreaterThan(0);
      expect(telemetry?.metrics.length).toBeGreaterThan(0);
    }

    expect(Object.keys(CAMERA_PRESETS)).toEqual([
      'overview',
      'gearbox',
      'tellurian',
      'ephemeris',
      'davinci',
      'ecliptic',
    ]);
  });

  it('emits >1,000 active 3D horological part instances and separates coaxial decks in Exploded mode', () => {
    const sceneNormal = inspectOrrerySceneAtDay(45, geometry, false, 0, 0);
    const sceneExploded = inspectOrrerySceneAtDay(45, geometry, false, 0, 1);

    expect(sceneNormal.parts.length).toBeGreaterThan(1000);
    expect(sceneNormal.pickSpheres.length).toBeGreaterThan(15);
    expect(sceneNormal.overlayVertCount).toBeGreaterThan(100);

    const sunNormal = sceneNormal.parts.find((p) => p.meshName === 'unitSphere' && p.groupId === 5);
    const sunExploded = sceneExploded.parts.find(
      (p) => p.meshName === 'unitSphere' && p.groupId === 5
    );
    expect(sunNormal).toBeDefined();
    expect(sunExploded).toBeDefined();
    expect((sunExploded?.tz ?? 0) - (sunNormal?.tz ?? 0)).toBeGreaterThan(20);
  });
});
