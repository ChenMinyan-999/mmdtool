"""Developer diagnostics: score layout, per-frame layers, worst frames."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmdance import choreo as C  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import synth as SY  # noqa: E402
from mmdance.mathx import quat_angle_between  # noqa: E402


def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else "examples/demo_song.wav"
    score = sys.argv[2] if len(sys.argv) > 2 else "out/score.json"
    a = MU.analyze(wav)
    ch = C.Choreo.load(score)
    print(f"{len(ch.slots)} slots")
    for s in ch.sorted_slots():
        print(f"  {s.beat:5.1f} -> {s.end:5.1f}  {s.move:14s} w={s.weight:.2f}  {s.note}")

    states = SY.bake(a, ch)
    print(f"\n{len(states)} frames")

    lows = []
    for st in states:
        lows.append((min(st.pos["toe.L"][1], st.pos["toe.R"][1]), st.frame))
    lows.sort()
    print("\nlowest toe frames:", [(int(f), round(float(h), 3)) for h, f in lows[:6]])

    worst = (0.0, 0, "")
    for i in range(1, len(states) - 1):
        for b in states[i].local:
            d1 = quat_angle_between(states[i - 1].local[b], states[i].local[b])
            d2 = quat_angle_between(states[i].local[b], states[i + 1].local[b])
            if abs(d2 - d1) > worst[0]:
                worst = (abs(d2 - d1), i, b)
    print(f"\nworst snap: {worst[0]:.1f} deg/frame^2 at frame {worst[1]} bone {worst[2]}")

    for f in range(max(0, worst[1] - 3), min(len(states), worst[1] + 4)):
        st = states[f]
        act = [
            f"{s.move}({(st.beat - s.beat) / s.beats:.2f},w{SY.slot_weight(st.beat, s.beat, s.beats, s.weight):.2f})"
            for s in ch.slots
            if SY.slot_weight(st.beat, s.beat, s.beats, s.weight) > 0.01
        ]
        print(f"  f{f:4d} beat {st.beat:6.2f} toeL {float(st.pos['toe.L'][1]):.3f} "
              f"ankL {float(st.pos['ankle.L'][1]):.3f} rootY {float(st.pos['root'][1]):.3f} | {' '.join(act)}")

    # per-frame arm.R euler trace around the snap
    print("\narm.R local quats around snap:")
    for f in range(max(0, worst[1] - 2), min(len(states), worst[1] + 3)):
        q = states[f].local.get("arm.R")
        print(f"  f{f}: {np.round(q, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
