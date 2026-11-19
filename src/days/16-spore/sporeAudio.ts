/**
 * Subtle Procedural WebAudio Tactile Acoustics for Day 16 — Spore.
 *
 * Synthesizes two organic physical micro-sounds with zero external assets:
 * 1. Dry papery endoperidium parchment crinkle when indenting/squeezing the sac.
 * 2. Soft low-frequency Helmholtz cavity pneumatic "phwuff" air burst on vortex-ring launch.
 *
 * Unlocked lazily on first user pointer gesture; fails silently if audio is unavailable.
 */
export class SporeAcoustics {
  private ctx: AudioContext | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private isDisposed = false;

  unlock(): void {
    if (this.isDisposed || typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtor) return;
      try {
        this.ctx = new AudioCtor();
        this.noiseBuffer = this.createSoftNoiseBuffer(this.ctx);
      } catch {
        this.ctx = null;
        return;
      }
    }
    if (this.ctx.state === 'suspended') {
      void this.ctx.resume().catch(() => {});
    }
  }

  private createSoftNoiseBuffer(ctx: AudioContext): AudioBuffer {
    const length = Math.floor(ctx.sampleRate * 1.2);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    // Warm pink/brown-tilted filtered noise for organic airflow & parchment texture
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    for (let i = 0; i < length; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.997 * b0 + white * 0.029;
      b1 = 0.985 * b1 + white * 0.032;
      b2 = 0.950 * b2 + white * 0.048;
      data[i] = Math.max(-1, Math.min(1, (b0 + b1 + b2 + white * 0.06) * 1.8));
    }
    return buffer;
  }

  playPuff(strength = 0.85): void {
    if (this.isDisposed || !this.ctx || !this.noiseBuffer || this.ctx.state !== 'running') return;
    const now = this.ctx.currentTime;
    const s = Math.max(0.25, Math.min(1.3, strength));

    // Layer 1: Pneumatic cavity "phwuff" (bandpass noise sweeping downward)
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(520 + s * 260, now);
    filter.frequency.exponentialRampToValueAtTime(165, now + 0.24 + s * 0.08);
    filter.Q.setValueAtTime(1.4, now);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.085 * s, now + 0.018);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.28 + s * 0.12);

    src.connect(filter);
    filter.connect(gain);
    gain.connect(this.ctx.destination);

    src.start(now, Math.random() * 0.5, 0.45);

    // Layer 2: Soft dry parchment flex transient (short crisp high-band crinkle)
    const crinkle = this.ctx.createBufferSource();
    crinkle.buffer = this.noiseBuffer;

    const hp = this.ctx.createBiquadFilter();
    hp.type = 'bandpass';
    hp.frequency.setValueAtTime(2200 + Math.random() * 450, now);
    hp.Q.setValueAtTime(2.8, now);

    const cGain = this.ctx.createGain();
    cGain.gain.setValueAtTime(0.0001, now);
    cGain.gain.linearRampToValueAtTime(0.024 * s, now + 0.008);
    cGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.065);

    crinkle.connect(hp);
    hp.connect(cGain);
    cGain.connect(this.ctx.destination);
    crinkle.start(now, Math.random() * 0.7, 0.09);
  }

  dispose(): void {
    this.isDisposed = true;
    if (this.ctx) {
      void this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    this.noiseBuffer = null;
  }
}
