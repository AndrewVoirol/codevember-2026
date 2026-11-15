import type { PigmentKey } from '../../core/palette';

export type { PigmentKey };

export const SEGMENTS_PER_BAND = 128;
export const TOTAL_BANDS = 200;
export const CORE_RADIUS = 0.36;

export type BandStatus = 'intact' | 'pulled' | 'snapping' | 'gone';

export interface BandSpec {
  id: number;
  layer: number;
  baseRadius: number;
  width: number;
  thickness: number;
  pigment: PigmentKey;
  restStrain: number;
  breakStrain: number;
  normal: [number, number, number];
  tangentU: [number, number, number];
  tangentV: [number, number, number];
}

export interface BandRuntime {
  id: number;
  status: BandStatus;
  currentStrain: number;
  pullU: number;
  pullAmount: number;
  pullDir: [number, number, number];
  twangPhase: number;
  twangAmp: number;
  snapU: number;
  leftArc: number;
  rightArc: number;
  leftVel: number;
  rightVel: number;
  curlAmount: number;
  fallOffset: [number, number, number];
  fallVel: [number, number, number];
  opacity: number;
}

export interface DustPuff {
  active: boolean;
  isSpeck: boolean;
  pos: [number, number, number];
  vel: [number, number, number];
  ageSec: number;
  maxAgeSec: number;
  size: number;
}

export interface SnapImpulseInfo {
  snapId: number;
  bandId: number;
  hitPos: [number, number, number];
  pullDir: [number, number, number];
  normal: [number, number, number];
  intensity: number;
}

export interface RubberBallDiagnostics {
  totalBands: number;
  intactCount: number;
  snappingCount: number;
  goneCount: number;
  avgStrain: number;
  ballRadius: number;
  activePullBandId: number | null;
  snapCount: number;
}
