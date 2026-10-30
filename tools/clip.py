"""Render a short clip (beat range) to GIF/MP4 - handy for sharing a moment."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from mmdance import choreo as C  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import render as RD  # noqa: E402
from mmdance import synth as SY  # noqa: E402


def main() -> int:
    wav = "examples/demo_song.wav"
    score = "out/score.json"
    beat0 = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    beat1 = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("out/clip.gif")

    a = MU.analyze(wav)
    ch = C.Choreo.load(score)
    states = SY.bake(a, ch)
    t0, t1 = beat0 * a.beat_period, beat1 * a.beat_period
    sel = [st for st in states if t0 - 1e-6 <= st.frame / SY.FPS <= t1]
    imgs = []
    for st in sel[::2]:  # 15 fps output
        sec = a.section_at_beat(max(0.0, st.beat))
        active = [s.move for s in ch.slots if SY.slot_weight(st.beat, s.beat, s.beats, s.weight) > 0.12]
        info = {
            "beat": st.beat,
            "section": sec.name,
            "energy": st.energy,
            "moves": " + ".join(active[:2]),
            "subtitle": f"demo_song.wav   {a.bpm:.1f} BPM",
        }
        imgs.append(RD.render_frame(st, info, size=(size, size)))
    out.parent.mkdir(parents=True, exist_ok=True)
    RD.make_gif(imgs, out, fps=15, colors=128, width=size)
    print(f"{out}  {out.stat().st_size/1024:.0f} KB  {len(imgs)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
