"""L0 - deterministic music analysis.

No ML: STFT -> spectral flux onset envelope -> autocorrelation tempo ->
phase-locked beat grid -> bar level RMS -> section segmentation.

This is deliberately dependency-free (stdlib ``wave`` + numpy).  It is exact
enough for a steady tempo track such as the bundled demo song.  For real music
you can swap this module for librosa / beat_this without touching anything
downstream: the contract is just :class:`MusicAnalysis`.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TARGET_SR = 22050
FRAME = 1024
HOP = 256
FPS = TARGET_SR / HOP  # ~86.1 fps


# --------------------------------------------------------------------------- #
# audio loading
# --------------------------------------------------------------------------- #
def _read_wav_pcm(path: Path):
    with wave.open(str(path), "rb") as fh:
        nch, sw, sr, nframes = fh.getnchannels(), fh.getsampwidth(), fh.getframerate(), fh.getnframes()
        raw = fh.readframes(nframes)
    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        data /= 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        data = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data.astype(np.float32), sr


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32.  Uses ffmpeg for non-wav input."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return _read_wav_pcm(path)
        except wave.Error:
            pass  # fall through to ffmpeg (float wav / extensible formats)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"cannot decode {path.suffix}: only PCM .wav works without ffmpeg on PATH"
        )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "conv.wav"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SR), str(tmp)],
            check=True,
        )
        return _read_wav_pcm(tmp)


def resample(x: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return x
    n_out = int(round(len(x) * target / sr))
    t_in = np.arange(len(x)) / sr
    t_out = np.arange(n_out) / target
    return np.interp(t_out, t_in, x).astype(np.float32)


# --------------------------------------------------------------------------- #
# analysis result
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    name: str
    start_beat: float
    end_beat: float
    energy: float

    @property
    def beats(self) -> float:
        return self.end_beat - self.start_beat


@dataclass
class MusicAnalysis:
    bpm: float
    beat_times: np.ndarray          # seconds, one entry per beat
    beat_energy: np.ndarray         # 0..1 accent strength per beat
    sections: list[Section]
    duration: float
    sr: int
    downbeat_offset: int = 0        # index into beat_times of the first downbeat
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_beats(self) -> int:
        return len(self.beat_times)

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    def beat_at(self, t: float) -> float:
        """Continuous beat index at time ``t`` (seconds), extrapolated outside."""
        if self.n_beats == 0:
            return 0.0
        t0 = float(self.beat_times[0])
        period = self.beat_period
        if t <= t0:
            return (t - t0) / period
        last = self.n_beats - 1
        if t >= self.beat_times[last]:
            return last + (t - self.beat_times[last]) / period
        return float(np.interp(t, self.beat_times, np.arange(self.n_beats)))

    def energy_at_beat(self, beat: float) -> float:
        i = int(np.clip(round(beat), 0, len(self.beat_energy) - 1))
        return float(self.beat_energy[i])

    def section_at_beat(self, beat: float) -> Section:
        for s in self.sections:
            if s.start_beat <= beat < s.end_beat:
                return s
        return self.sections[-1]

    def to_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 3),
            "duration": round(self.duration, 3),
            "beats": self.n_beats,
            "bar_length": 4,
            "downbeat_offset": self.downbeat_offset,
            "sections": [
                {
                    "name": s.name,
                    "start_beat": round(s.start_beat, 2),
                    "end_beat": round(s.end_beat, 2),
                    "energy": round(s.energy, 3),
                }
                for s in self.sections
            ],
            "beat_energy": [round(float(e), 3) for e in self.beat_energy],
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# core analysis
# --------------------------------------------------------------------------- #
def _stft_mag(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    win = np.hanning(FRAME).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP]
    spec = np.fft.rfft(frames * win, axis=1)
    return np.abs(spec).astype(np.float32)


def _freqs(sr: int) -> np.ndarray:
    return np.fft.rfftfreq(FRAME, 1.0 / sr)


def onset_envelope(mag: np.ndarray, sr: int):
    """Log-magnitude spectral flux + band wise envelopes."""
    freqs = _freqs(sr)
    logmag = np.log1p(mag * 10.0)
    diff = np.diff(logmag, axis=0, prepend=logmag[:1])
    flux = np.maximum(diff, 0.0).sum(axis=1)

    def band(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return mag[:, sel].sum(axis=1) if sel.any() else np.zeros(len(mag))

    low = band(30, 160)
    high = band(6000, 11000)

    def norm(v):
        v = v - np.convolve(v, np.ones(15) / 15.0, mode="same")  # remove slow drift
        s = np.std(v)
        return (v / s) if s > 1e-9 else np.zeros_like(v)

    return norm(flux), norm(low), norm(high)


def _tempo_from_acf(env: np.ndarray, fps: float, lo: float = 60.0, hi: float = 190.0) -> float:
    e = env - env.mean()
    if not np.any(e):
        return 120.0
    acf = np.correlate(e, e, mode="full")[len(e) - 1 :]
    acf = acf / (acf[0] + 1e-9)
    min_lag = max(1, int(round(fps * 60.0 / hi)))
    max_lag = min(len(acf) - 2, int(round(fps * 60.0 / lo)))
    lags = np.arange(min_lag, max_lag + 1)
    if len(lags) < 3:
        return 120.0
    bpm = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.55) ** 2)
    score = acf[lags] * prior
    # parabolic refinement around the best integer lag
    i = int(np.argmax(score))
    lag = float(lags[i])
    if 0 < i < len(lags) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            lag += 0.5 * (y0 - y2) / denom
    return float(60.0 * fps / lag)


def _beat_phase(env: np.ndarray, period: float) -> float:
    """Phase (in envelope frames) that puts the grid on the strongest onsets."""
    n = len(env)
    best_phase, best_score = 0.0, -1e18
    for phase in np.arange(0.0, period, 0.25):
        pos = np.arange(phase, n - 1, period)
        if len(pos) < 2:
            continue
        idx = pos.astype(int)
        frac = pos - idx
        vals = env[idx] * (1 - frac) + env[idx + 1] * frac
        score = vals.mean() - 0.5 * vals.std()
        if score > best_score:
            best_score, best_phase = score, phase
    return float(best_phase)


def analyze(path: str | Path, bpm_hint: float | None = None) -> MusicAnalysis:
    x, sr = load_audio(path)
    x = resample(x, sr, TARGET_SR)
    sr = TARGET_SR
    duration = len(x) / sr

    mag = _stft_mag(x)
    flux, low, high = onset_envelope(mag, sr)

    bpm = float(bpm_hint) if bpm_hint else _tempo_from_acf(flux, FPS)
    period = FPS * 60.0 / bpm  # in envelope frames

    # refine period + phase jointly (handles small tempo drift / estimation error)
    best = (None, -1e18, 0.0)
    for scale in np.linspace(0.985, 1.015, 13):
        p = period * scale
        ph = _beat_phase(flux, p)
        pos = np.arange(ph, len(flux) - 1, p)
        if len(pos) < 4:
            continue
        idx = pos.astype(int)
        frac = pos - idx
        vals = flux[idx] * (1 - frac) + flux[idx + 1] * frac
        score = vals.mean() - 0.5 * vals.std()
        if score > best[1]:
            best = (p, score, ph)
    period, _, phase = best
    if period is None:
        period, phase = FPS * 60.0 / bpm, 0.0

    # envelope frame i covers samples [i*HOP, i*HOP+FRAME); the flux peak of an
    # onset therefore appears about half a window early -> compensate.
    half_window = 0.5 * FRAME / sr

    # generate a grid that covers the whole track (including the beat just
    # before t=0, so that a leading downbeat is not lost), keep [0, duration].
    beat_period_s = 60.0 / bpm
    k_lo = int(np.floor((-1.05 * beat_period_s - half_window) / (period * HOP / sr))) - 1
    k_hi = int(np.ceil((len(flux) - phase) / period)) + 1
    frame_pos_all = phase + np.arange(k_lo, k_hi) * period
    times_all = frame_pos_all * HOP / sr + half_window
    keep = (times_all >= -1.05 * beat_period_s) & (times_all <= duration)
    frame_pos = frame_pos_all[keep]
    beat_times = times_all[keep]
    n_beats = len(beat_times)

    # per-beat accent strength (peak flux in a +-60ms window around the beat)
    w = max(1, int(0.06 * sr / HOP))
    strengths = []
    for fp in frame_pos:
        i = int(round(fp))
        lo, hi = max(0, i - w), min(len(flux), i + w + 1)
        strengths.append(float(flux[lo:hi].max()) if hi > lo else 0.0)
    strengths = np.asarray(strengths)
    smax = np.percentile(strengths, 90) if len(strengths) else 1.0
    beat_energy = np.clip(strengths / (smax + 1e-9), 0.0, 1.0)

    # downbeat: most tracks / edits begin on a bar line, so anchor to the beat
    # closest to t=0 when there is one.  Otherwise fall back to beat-sync
    # spectral novelty (chord changes usually land on bar lines).
    best_off = 0
    near_zero = int(np.argmin(np.abs(beat_times)))
    if abs(beat_times[near_zero]) <= 0.2 * beat_period_s:
        best_off = near_zero % 4
    else:
        beat_vec = _beat_spectra(mag, frame_pos, HOP)
        novelty = np.zeros(len(frame_pos))
        if len(frame_pos) > 1:
            novelty[1:] = np.linalg.norm(np.diff(beat_vec, axis=0), axis=1)
        best_val = -1e18
        for off in range(4):
            idx = np.arange(off, n_beats, 4)
            if len(idx) == 0:
                continue
            val = float(novelty[idx].mean())
            if val > best_val:
                best_val, best_off = val, off

    # rotate the grid so that beat 0 is a downbeat -> bars are [0,4,8,...]
    if best_off:
        beat_times = beat_times[best_off:]
        beat_energy = beat_energy[best_off:]
        frame_pos = frame_pos[best_off:]
        n_beats = len(beat_times)

    # sections: bar level RMS energy
    bar_len = 4
    n_bars = max(1, n_beats // bar_len)
    bar_rms = np.zeros(n_bars)
    for b in range(n_bars):
        t0 = beat_times[b * bar_len]
        t1 = beat_times[min(n_beats - 1, (b + 1) * bar_len)] + 60.0 / bpm
        i0, i1 = int(t0 * sr), int(min(len(x), t1 * sr))
        seg = x[i0:i1]
        bar_rms[b] = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
    if bar_rms.max() > 0:
        bar_rms = bar_rms / (np.percentile(bar_rms, 95) + 1e-9)
    bar_rms = np.clip(bar_rms, 0.0, 1.2)
    # smooth over 2 bars
    if n_bars > 2:
        k = np.array([0.25, 0.5, 0.25])
        bar_rms = np.convolve(bar_rms, k, mode="same")

    sections = _segment(bar_rms, bar_len, n_beats)
    return MusicAnalysis(
        bpm=bpm,
        beat_times=beat_times,
        beat_energy=beat_energy,
        sections=sections,
        duration=duration,
        sr=sr,
        downbeat_offset=0,
        source=str(path),
        extra={"bar_energy": bar_rms.tolist()},
    )


def _label(energy: float, index: int, count: int) -> str:
    if energy < 0.5:
        if index == 0:
            return "intro"
        if index == count - 1:
            return "outro"
        return "break"
    if energy >= 0.78:
        return "chorus"
    return "verse"


def _beat_spectra(mag: np.ndarray, frame_pos: np.ndarray, hop: int) -> np.ndarray:
    """Beat-synchronous log spectrum vectors (one row per beat), L2 normalised."""
    if len(frame_pos) == 0:
        return np.zeros((0, mag.shape[1]))
    edges = np.concatenate([[frame_pos[0] - 0.5 * (frame_pos[1] - frame_pos[0]) if len(frame_pos) > 1 else frame_pos[0]],
                            frame_pos[1:] - 0.5 * np.diff(frame_pos),
                            [frame_pos[-1] + 0.5 * (frame_pos[1] - frame_pos[0]) if len(frame_pos) > 1 else frame_pos[-1]]])
    vecs = np.zeros((len(frame_pos), mag.shape[1]), dtype=np.float32)
    for i in range(len(frame_pos)):
        lo = int(np.clip(np.floor(edges[i]), 0, len(mag) - 1))
        hi = int(np.clip(np.ceil(edges[i + 1]), lo + 1, len(mag)))
        seg = mag[lo:hi]
        vecs[i] = np.log1p(seg.mean(axis=0) * 10.0)
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (n + 1e-9)


def _segment(bar_rms: np.ndarray, bar_len: int, n_beats: int) -> list[Section]:
    """Split at bar-level energy change points, then name sections by loudness."""
    n_bars = len(bar_rms)
    if n_bars == 0:
        return [Section("full", 0.0, float(n_beats), 0.6)]

    diffs = np.abs(np.diff(bar_rms)) if n_bars > 1 else np.array([])
    thr = max(0.10, 0.6 * float(np.std(bar_rms))) if n_bars > 2 else 1e9
    boundaries = [0]
    for i, d in enumerate(diffs, start=1):
        if d >= thr and i - boundaries[-1] >= 2:
            boundaries.append(i)
    if boundaries[-1] != n_bars:
        boundaries.append(n_bars)

    sections: list[Section] = []
    for idx in range(len(boundaries) - 1):
        b0, b1 = boundaries[idx], boundaries[idx + 1]
        if b1 <= b0:
            continue
        e = float(np.mean(bar_rms[b0:b1]))
        sections.append(Section("", float(b0 * bar_len), float(min(b1 * bar_len, n_beats)), min(1.0, max(0.0, e))))
    for i, s in enumerate(sections):
        s.name = _label(s.energy, i, len(sections))
    if len(sections) > 1:
        sections[0].end_beat = min(sections[0].end_beat, n_beats)
    sections[-1].end_beat = float(n_beats)
    return sections
