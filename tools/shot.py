"""Render a handful of single frames for visual inspection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmdance import choreo as C  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import render as RD  # noqa: E402
from mmdance import synth as SY  # noqa: E402


def main() -> int:
    wav = "examples/demo_song.wav"
    score = "out/score.json"
    frames = [int(x) for x in (sys.argv[1:] or ["120", "300", "420"])]
    a = MU.analyze(wav)
    ch = C.Choreo.load(score)
    states = SY.bake(a, ch)
    out = Path("out_shots")
    out.mkdir(exist_ok=True)
    for f in frames:
        f = min(f, len(states) - 1)
        st = states[f]
        sec = a.section_at_beat(max(0.0, st.beat))
        active = [s.move for s in ch.slots if SY.slot_weight(st.beat, s.beat, s.beats, s.weight) > 0.12]
        info = {
            "beat": st.beat,
            "section": sec.name,
            "energy": st.energy,
            "moves": " + ".join(active[:2]),
            "subtitle": "demo_song.wav   120.0 BPM",
        }
        img = RD.render_frame(st, info)
        p = out / f"frame_{f:04d}.png"
        img.save(p)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
