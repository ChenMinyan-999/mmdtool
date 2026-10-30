"""Render a horizontal film strip of consecutive frames (timing inspection)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from mmdance import choreo as C  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import render as RD  # noqa: E402
from mmdance import synth as SY  # noqa: E402


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    a = MU.analyze("examples/demo_song.wav")
    ch = C.Choreo.load("out/score.json")
    states = SY.bake(a, ch)
    tiles = []
    for i in range(count):
        st = states[min(start + i * step, len(states) - 1)]
        info = {"beat": st.beat, "section": a.section_at_beat(max(0, st.beat)).name, "energy": st.energy, "moves": "", "subtitle": ""}
        tiles.append(RD.render_frame(st, info, size=(360, 360)))
    w = 360 * count
    sheet = Image.new("RGB", (w, 380), (12, 12, 20))
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        sheet.paste(t.convert("RGB"), (i * 360, 0))
        d.text((i * 360 + 8, 364), f"f{start + i * step}", font=RD._font(14), fill=(200, 200, 220))
    out = Path(f"out_strip_{start}_{count}x{step}.png")
    sheet.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
