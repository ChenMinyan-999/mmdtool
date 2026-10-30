"""Command line entry point.

    python -m mmdance.cli demo                 # full pipeline on the demo song
    python -m mmdance.cli build --wav song.wav # any track
    python -m mmdance.cli plan  --wav song.wav # only the choreography score
    python -m mmdance.cli probe --move clap    # joint positions of one move
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import choreo as C
from . import moves as MV
from . import music as MU
from . import render as RD
from . import synth as SY
from . import validate as VD
from .mathx import set_mmd_axis
from .vmd import VmdMotion

ROOT = Path(__file__).resolve().parent.parent
DEMO_WAV = ROOT / "examples" / "demo_song.wav"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _labels(states, analysis, choreo):
    out = []
    for st in states:
        beat = st.beat
        sec = analysis.section_at_beat(max(0.0, beat))
        active = []
        for s in choreo.slots:
            w = SY.slot_weight(beat, s.beat, s.beats, s.weight)
            if w > 0.12:
                active.append(s.move)
        out.append(
            {
                "beat": beat,
                "section": sec.name,
                "energy": st.energy,
                "moves": " + ".join(active[:3]),
                "subtitle": f"{Path(analysis.source).name}   {analysis.bpm:.1f} BPM",
            }
        )
    return out


def build(
    wav: Path,
    out_dir: Path,
    score_path: Path | None = None,
    style: str = "idol-pop",
    difficulty: float = 0.6,
    seed: int = 7,
    planner: str = "rule",
    bpm: float | None = None,
    seconds: float | None = None,
    do_render: bool = True,
    size: tuple[int, int] = (720, 720),
    gif: bool = True,
    flip_z: bool = True,
    no_ik: bool = False,
    keep_frames: bool = False,
) -> dict:
    t0 = time.time()
    set_mmd_axis(flip_z)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- L0 ------------------------------------------------------------
    _log(f"[L0] analysing {wav.name} ...")
    analysis = MU.analyze(wav, bpm_hint=bpm)
    dur = seconds if seconds else analysis.duration
    (out_dir / "music_analysis.json").write_text(
        json.dumps({**analysis.to_dict(), "duration": dur}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        f"     bpm={analysis.bpm:.2f}  beats={analysis.n_beats}  "
        f"sections={[ (s.name, round(s.energy,2)) for s in analysis.sections ]}"
    )

    # ---- L1 ------------------------------------------------------------
    if score_path and score_path.exists():
        _log(f"[L1] loading score {score_path.name}")
        choreo = C.Choreo.load(score_path)
    elif planner == "llm":
        _log("[L1] asking the LLM for a choreography score ...")
        choreo = C.plan_with_llm(analysis, style=style, difficulty=difficulty)
    else:
        _log("[L1] rule planner ...")
        choreo = C.plan(analysis, style=style, difficulty=difficulty, seed=seed)
    errors, warnings = C.validate(choreo)
    for w in warnings[:6]:
        _log(f"     warn: {w}")
    if errors:
        for e in errors[:10]:
            _log(f"     ERROR: {e}")
        raise SystemExit("choreography score is invalid")
    choreo.save(out_dir / "score.json")
    _log(f"     {len(choreo.slots)} slots written to score.json")

    # ---- L2 ------------------------------------------------------------
    _log("[L2] synthesising (groove + moves + foot lock IK) ...")
    max_frames = int(round(seconds * SY.FPS)) if seconds else None
    states = SY.bake(analysis, choreo, max_frames=max_frames)
    _log(f"     {len(states)} frames @ {SY.FPS} fps")

    # ---- L3 ------------------------------------------------------------
    _log("[L3] baking VMD ...")
    motion = SY.to_vmd(states, model_name="mmdance", emit_ik=not no_ik)
    vmd_path = out_dir / "motion.vmd"
    size_bytes = motion.save(vmd_path)
    _log(f"     {vmd_path.name}  {size_bytes/1024:.0f} KB  {motion.summary()}")

    # read the file back: whatever MMD reads is what we validate
    vmd_check = VD.check_vmd(VmdMotion.load(vmd_path))
    _log(f"     vmd structure: {vmd_check['status']}  bones={vmd_check['bones']} keys={vmd_check['keys']}")

    # ---- L4 ------------------------------------------------------------
    _log("[L4] verification ...")
    report = VD.check(states, choreo=choreo, analysis=analysis)
    report["vmd"] = vmd_check
    if vmd_check["status"] != "PASS":
        report["overall"] = "FAIL"
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(VD.format_report(report))

    # ---- preview -------------------------------------------------------
    if do_render:
        _log("[render] drawing preview ...")
        labels = _labels(states, analysis, choreo)
        frames_dir = out_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        paths = RD.render_sequence(states, labels, frames_dir, size=size)
        _log(f"     {len(paths)} frames")

        mp4 = out_dir / "preview.mp4"
        if RD.encode_mp4(frames_dir, mp4):
            _log(f"     {mp4.name}  {mp4.stat().st_size/1024:.0f} KB")
        if gif:
            from PIL import Image

            step = max(1, round(SY.FPS / 15))
            imgs = [Image.open(p) for p in paths[::step]]
            gif_path = out_dir / "preview.gif"
            RD.make_gif(imgs, gif_path)
            _log(f"     {gif_path.name}  {gif_path.stat().st_size/1024:.0f} KB")

        # contact sheet at the "peak" of a spread of slots (more informative
        # than uniform sampling, which mostly catches transitions)
        keys = _sheet_indices(states, choreo, count=9)
        from PIL import Image

        imgs = [RD.render_frame(states[i], labels[i], size=size) for i in keys]
        titles = [f"beat {states[i].beat:4.1f}  {labels[i]['moves'][:34]}" for i in keys]
        RD.contact_sheet(imgs, titles, out_dir / "contact_sheet.png")
        _log("     contact_sheet.png")
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    _log(f"[done] {time.time() - t0:.1f}s -> {out_dir}")
    return {"report": report, "vmd": str(vmd_path), "choreo": choreo.to_dict()}


def _sheet_indices(states, choreo=None, count: int = 9) -> list[int]:
    n = len(states)
    if n <= 2:
        return list(range(n))
    slots = choreo.sorted_slots() if choreo is not None else []
    if len(slots) < 3:
        return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]
    k = min(count, len(slots))
    picks = [slots[int(round(i * (len(slots) - 1) / (k - 1)))] for i in range(k)] if k > 1 else slots
    out: list[int] = []
    for s in picks:
        beat = s.beat + 0.42 * s.beats
        i = min(range(n), key=lambda j: abs(states[j].beat - beat))
        if i not in out:
            out.append(i)
    return out[:count]


# --------------------------------------------------------------------------- #
def cmd_demo(args) -> int:
    wav = Path(args.wav) if args.wav else DEMO_WAV
    if not wav.exists():
        _log(f"demo song missing, generating {wav} ...")
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_music.py"), str(wav)], check=True)
    build(
        wav=wav,
        out_dir=Path(args.out),
        style=args.style,
        difficulty=args.difficulty,
        seed=args.seed,
        planner=args.planner,
        bpm=args.bpm,
        seconds=args.seconds,
        do_render=not args.no_render,
        gif=not args.no_gif,
        flip_z=not args.no_flip,
        no_ik=getattr(args, "no_ik", False),
        keep_frames=getattr(args, "keep_frames", False),
    )
    return 0


def cmd_build(args) -> int:
    build(
        wav=Path(args.wav),
        out_dir=Path(args.out),
        score_path=Path(args.score) if args.score else None,
        style=args.style,
        difficulty=args.difficulty,
        seed=args.seed,
        planner=args.planner,
        bpm=args.bpm,
        seconds=args.seconds,
        do_render=not args.no_render,
        gif=not args.no_gif,
        flip_z=not args.no_flip,
        no_ik=getattr(args, "no_ik", False),
        keep_frames=getattr(args, "keep_frames", False),
    )
    return 0


def cmd_plan(args) -> int:
    analysis = MU.analyze(Path(args.wav), bpm_hint=args.bpm)
    choreo = C.plan(analysis, style=args.style, difficulty=args.difficulty, seed=args.seed)
    errors, warnings = C.validate(choreo)
    out = Path(args.out)
    choreo.save(out)
    _log(f"score -> {out}  ({len(choreo.slots)} slots, {len(warnings)} warnings, {len(errors)} errors)")
    for w in warnings:
        _log("  warn: " + w)
    for e in errors:
        _log("  ERROR: " + e)
    return 1 if errors else 0


def cmd_probe(args) -> int:
    """Print world joint positions of a move at several phases (tuning aid)."""
    mv = MV.get(args.move)
    base = {}
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        pose = mv.pose_at(p)
        local = {j: __import__("mmdance.mathx", fromlist=["quat_from_euler_deg"]).quat_from_euler_deg(*v) for j, v in pose.items()}
        st = SY.fk(local)
        row = {
            "phase": p,
            "hand.L": [round(float(x), 2) for x in st.pos["wrist.L"]],
            "hand.R": [round(float(x), 2) for x in st.pos["wrist.R"]],
            "foot.L": [round(float(x), 2) for x in st.pos["ankle.L"]],
            "hip": [round(float(x), 2) for x in st.pos["root"]],
        }
        print(row)
        base[p] = row
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mmdance", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, wav_required: bool):
        if wav_required:
            p.add_argument("--wav", required=True)
        else:
            p.add_argument("--wav", default=None)
        p.add_argument("--out", default="out")
        p.add_argument("--style", default="idol-pop")
        p.add_argument("--difficulty", type=float, default=0.6)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--bpm", type=float, default=None)
        p.add_argument("--seconds", type=float, default=None)
        p.add_argument("--planner", choices=["rule", "llm"], default="rule")
        p.add_argument("--no-render", action="store_true")
        p.add_argument("--no-gif", action="store_true")
        p.add_argument("--no-flip", action="store_true", help="disable MMD left-handed Z flip")
        p.add_argument("--no-ik", action="store_true", help="omit leg IK bone keyframes")
        p.add_argument("--keep-frames", action="store_true", help="keep the rendered PNG sequence")

    p_demo = sub.add_parser("demo", help="run the bundled demo end to end")
    common(p_demo, False)
    p_demo.set_defaults(func=cmd_demo)

    p_build = sub.add_parser("build", help="build a VMD from any audio file")
    common(p_build, True)
    p_build.add_argument("--score", default=None, help="reuse an existing choreography score")
    p_build.set_defaults(func=cmd_build)

    p_plan = sub.add_parser("plan", help="only produce the choreography score")
    p_plan.add_argument("--wav", required=True)
    p_plan.add_argument("--out", default="out/score.json")
    p_plan.add_argument("--style", default="idol-pop")
    p_plan.add_argument("--difficulty", type=float, default=0.6)
    p_plan.add_argument("--seed", type=int, default=7)
    p_plan.add_argument("--bpm", type=float, default=None)
    p_plan.set_defaults(func=cmd_plan)

    p_probe = sub.add_parser("probe", help="inspect a move's joint positions")
    p_probe.add_argument("--move", required=True)
    p_probe.set_defaults(func=cmd_probe)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
