"""Render only the contact sheet (fast visual iteration)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmdance import choreo as C  # noqa: E402
from mmdance import cli as CLI  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import render as RD  # noqa: E402
from mmdance import synth as SY  # noqa: E402


def main() -> int:
    wav = "examples/demo_song.wav"
    a = MU.analyze(wav)
    ch = C.Choreo.load("out/score.json")
    states = SY.bake(a, ch)
    idx = CLI._sheet_indices(states, ch, count=9)
    imgs, titles = [], []
    for i in idx:
        st = states[i]
        sec = a.section_at_beat(max(0.0, st.beat))
        active = [s.move for s in ch.slots if SY.slot_weight(st.beat, s.beat, s.beats, s.weight) > 0.12]
        info = {"beat": st.beat, "section": sec.name, "energy": st.energy, "moves": " + ".join(active[:2]), "subtitle": "demo_song.wav   120.0 BPM"}
        imgs.append(RD.render_frame(st, info))
        titles.append(f"beat {st.beat:4.1f}  {' + '.join(active[:2])[:34]}")
    out = Path("out_sheet.png")
    RD.contact_sheet(imgs, titles, out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
