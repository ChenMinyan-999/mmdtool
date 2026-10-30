"""Synthesise the bundled demo song (no external samples, no licence issues).

120 BPM, 4/4, 12 bars (24 s): intro -> groove with 4-on-the-floor kick ->
chorus with an extra chord stab layer -> outro.  Strong downbeat accents are
deliberate: it gives the L0 analysis something unambiguous to lock onto.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 12


def env_exp(n: int, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.exp(-t / decay)


def kick(dur: float = 0.45) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 150.0 * np.exp(-t / 0.035) + 45.0
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * env_exp(n, 0.10)
    click = np.random.default_rng(1).normal(0, 1, n) * env_exp(n, 0.004) * 0.25
    return np.tanh((body + click) * 1.5)


def snare(dur: float = 0.28) -> np.ndarray:
    n = int(dur * SR)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, n)
    noise = np.convolve(noise, np.array([1, -0.85]), mode="same")  # brighten
    tone = np.sin(2 * np.pi * 185 * np.arange(n) / SR) * env_exp(n, 0.05)
    return (noise * env_exp(n, 0.075) * 0.7 + tone * 0.5)


def hat(dur: float = 0.07) -> np.ndarray:
    n = int(dur * SR)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, n)
    noise = np.diff(noise, prepend=0.0)  # crude high pass
    return noise * env_exp(n, 0.018) * 0.5


def bass(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    wave_ = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5 + np.sin(2 * np.pi * freq * 2 * t) * 0.25
    # one pole low pass
    out = np.zeros(n)
    a = 0.22
    prev = 0.0
    for i in range(n):
        prev = prev * (1 - a) + wave_[i] * a
        out[i] = prev
    return out * env_exp(n, 0.18)


def stab(freqs, dur: float = 0.30) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        out += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.003 * t)
    return out / (len(freqs) * 1.6) * env_exp(n, 0.09)


def note(name: str) -> float:
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    pitch, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[pitch] + (octave - 4) * 12 - 9) / 12.0)


CHORDS = {
    "Am": (["A2", "A3", "C4", "E4"], "A2"),
    "F": (["F2", "F3", "A3", "C4"], "F2"),
    "C": (["C3", "C4", "E4", "G4"], "C3"),
    "G": (["G2", "G3", "B3", "D4"], "G2"),
}
PROGRESSION = ["Am", "F", "C", "G"] * 3  # 12 bars


def build() -> np.ndarray:
    n_total = int(BARS * BAR * SR)
    mix = np.zeros(n_total)

    def put(sig: np.ndarray, t: float, gain: float = 1.0) -> None:
        i = int(t * SR)
        j = min(n_total, i + len(sig))
        if i < n_total:
            mix[i:j] += sig[: j - i] * gain

    k, s, h = kick(), snare(), hat()
    for bar in range(BARS):
        chord = PROGRESSION[bar]
        notes, root = CHORDS[chord]
        bar_t = bar * BAR
        intro = bar < 2
        outro = bar >= 10

        # drums
        for b in range(4):
            t = bar_t + b * BEAT
            if outro:
                if b == 0:
                    put(k, t, 0.9)
            elif intro:
                if b in (0, 2):
                    put(k, t, 0.8)
            else:
                put(k, t, 1.0)
                if b in (1, 3):
                    put(s, t, 0.7)
        for e in range(8):
            t = bar_t + e * BEAT / 2
            if not outro:
                put(h, t, 0.22 if e % 2 else 0.3)

        # bass: 8th notes, root with an octave hop
        if not outro:
            for e in range(8):
                t = bar_t + e * BEAT / 2
                f = note(root) * (2 if e in (3, 7) else 1)
                put(bass(f, BEAT / 2 * 0.95), t, 0.5 if intro else 0.75)

        # chord stabs: bar start, and 3& in the chorus
        freqs = [note(x) for x in notes]
        put(stab(freqs), bar_t, 0.5 if intro else 0.7)
        if not intro and not outro:
            put(stab(freqs), bar_t + 2.5 * BEAT, 0.55)
            put(stab(freqs, 0.18), bar_t + 3.5 * BEAT, 0.4)

    # gentle limiter + fade
    mix = np.tanh(mix * 1.2)
    fade = int(0.4 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix /= max(1e-9, np.max(np.abs(mix))) * 1.05

    # fake stereo: haas delay on the hats-ish content
    delay = int(0.004 * SR)
    left = mix.copy()
    right = np.concatenate([np.zeros(delay), mix[:-delay]]) * 0.15 + mix * 0.85
    return np.stack([left, right], axis=1)


def save(path: Path, stereo: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(stereo, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/demo_song.wav")
    save(out, build())
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {BARS * BAR:.1f}s, {BPM:.0f} BPM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
