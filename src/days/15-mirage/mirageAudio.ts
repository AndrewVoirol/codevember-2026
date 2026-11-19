export interface MirageAudioSynth {
  unlock: () => void;
  playEmberSpark: (intensity: number) => void;
  playPaperShift: (intensity: number) => void;
  dispose: () => void;
}

/**
 * Zero-chrome procedural WebAudio synthesizer for warm ember crackles
 * and subtle cotton-rag paper rustles on scroll/click.
 */
export function createMirageAudio(): MirageAudioSynth {
  let ctx: AudioContext | null = null;
  let master: GainNode | null = null;
  let noiseBuf: AudioBuffer | null = null;
  let lastShiftMs = 0;

  function ensure(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    const AudioCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtor) return null;
    if (!ctx) {
      try {
        ctx = new AudioCtor();
        master = ctx.createGain();
        master.gain.setValueAtTime(0.32, ctx.currentTime);
        master.connect(ctx.destination);

        const len = Math.floor(ctx.sampleRate * 0.14);
        noiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
        const ch = noiseBuf.getChannelData(0);
        for (let i = 0; i < len; i++) {
          ch[i] = Math.random() * 2 - 1;
        }
      } catch {
        ctx = null;
        master = null;
        return null;
      }
    }
    if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => {});
    }
    return ctx;
  }

  return {
    unlock: () => {
      ensure();
    },
    playEmberSpark: (intensity: number) => {
      const ac = ensure();
      if (!ac || !master || !noiseBuf) return;
      const now = ac.currentTime;
      const amp = Math.min(0.85, Math.max(0.25, intensity));

      // Crisp dry paper-fold flutter + ember crackle
      const src = ac.createBufferSource();
      src.buffer = noiseBuf;
      const bp = ac.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.setValueAtTime(2400, now);
      bp.frequency.exponentialRampToValueAtTime(1100, now + 0.12);
      bp.Q.setValueAtTime(2.6, now);

      const g = ac.createGain();
      g.gain.setValueAtTime(0.001, now);
      g.gain.exponentialRampToValueAtTime(0.18 * amp, now + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0005, now + 0.13);

      src.connect(bp);
      bp.connect(g);
      g.connect(master);
      src.start(now);
      src.stop(now + 0.14);

      // Warm harmonic ember bell resonance
      const osc = ac.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(420, now);
      osc.frequency.exponentialRampToValueAtTime(630, now + 0.18);
      const og = ac.createGain();
      og.gain.setValueAtTime(0.001, now);
      og.gain.exponentialRampToValueAtTime(0.055 * amp, now + 0.02);
      og.gain.exponentialRampToValueAtTime(0.0004, now + 0.24);
      osc.connect(og);
      og.connect(master);
      osc.start(now);
      osc.stop(now + 0.25);
    },
    playPaperShift: (intensity: number) => {
      const nowWall = performance.now();
      if (nowWall - lastShiftMs < 95) return;
      lastShiftMs = nowWall;

      const ac = ensure();
      if (!ac || !master || !noiseBuf) return;
      const now = ac.currentTime;
      const amp = Math.min(0.45, Math.max(0.06, intensity));

      const src = ac.createBufferSource();
      src.buffer = noiseBuf;
      const bp = ac.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.setValueAtTime(1450, now);
      bp.Q.setValueAtTime(1.4, now);

      const g = ac.createGain();
      g.gain.setValueAtTime(0.001, now);
      g.gain.exponentialRampToValueAtTime(0.05 * amp, now + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0004, now + 0.10);

      src.connect(bp);
      bp.connect(g);
      g.connect(master);
      src.start(now);
      src.stop(now + 0.11);
    },
    dispose: () => {
      if (ctx) {
        void ctx.close().catch(() => {});
        ctx = null;
        master = null;
        noiseBuf = null;
      }
    },
  };
}
