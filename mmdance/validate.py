"""L4 - automatic acceptance checks.

Nobody should have to scrub a three minute VMD by hand.  These metrics run on
the baked frames and produce a machine readable report plus a PASS/WARN/FAIL
line per check:

  foot slide       planted foot must not skate across the floor
  penetration      no foot below the floor plane
  balance          centre of mass over the support polygon
  joint limits     nothing bent past a human range
  beat alignment   motion accents land on the beat
  smoothness       no per-frame snapping
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mathx import quat_angle_between
from .rig import UNIT_CM
from .synth import PoseState, compose_pose

FPS = 30.0

#: rough segment mass fractions (Dempster), keyed by a joint name
MASSES = {
    "head": 0.08,
    "chest": 0.30,
    "root": 0.14,
    "shoulder": 0.03,
    "elbow": 0.02,
    "leg": 0.10,
    "knee": 0.045,
    "ankle": 0.015,
}


@dataclass
class Metric:
    name: str
    value: float
    unit: str
    status: str  # PASS / WARN / FAIL
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "unit": self.unit,
            "status": self.status,
            "detail": self.detail,
        }


def _status(value: float, warn: float, fail: float) -> str:
    if value >= fail:
        return "FAIL"
    if value >= warn:
        return "WARN"
    return "PASS"


def centre_of_mass(st: PoseState) -> np.ndarray:
    total = 0.0
    acc = np.zeros(3)
    pairs = [
        ("head", st.pos["head"] + np.array([0, 1.3, 0]), MASSES["head"]),
        ("chest", st.pos["chest"], MASSES["chest"]),
        ("root", st.pos["root"], MASSES["root"]),
    ]
    for side in ("L", "R"):
        pairs += [
            ("shoulder", (st.pos[f"shoulder.{side}"] + st.pos[f"arm.{side}"]) / 2, MASSES["shoulder"]),
            ("elbow", (st.pos[f"elbow.{side}"] + st.pos[f"wrist.{side}"]) / 2, MASSES["elbow"]),
            ("leg", (st.pos[f"leg.{side}"] + st.pos[f"knee.{side}"]) / 2, MASSES["leg"]),
            ("knee", (st.pos[f"knee.{side}"] + st.pos[f"ankle.{side}"]) / 2, MASSES["knee"]),
            ("ankle", st.pos[f"ankle.{side}"], MASSES["ankle"]),
        ]
    for _, p, m in pairs:
        acc += np.asarray(p) * m
        total += m
    return acc / total


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _point_margin(p: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    """Signed distance from ``p`` to the hull boundary (positive = inside)."""
    if len(hull) < 3:
        if not hull:
            return -99.0
        return -min(math.hypot(p[0] - h[0], p[1] - h[1]) for h in hull)
    inside = True
    best = 1e9
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        px, py = p[0] - a[0], p[1] - a[1]
        c = ex * py - ey * px
        if c < 0:
            inside = False
        best = min(best, abs(c) / (math.hypot(ex, ey) + 1e-9))
    return best if inside else -best


def contact_points(st: PoseState) -> list[tuple[float, float]]:
    """Ground contact polygon corners (x, z) of whatever is touching the floor."""
    pts = []
    for side in ("L", "R"):
        ankle = st.pos[f"ankle.{side}"]
        toe = st.pos[f"toe.{side}"]
        if toe[1] < 0.55 or ankle[1] < 0.95:
            for base in (ankle, toe):
                for dx in (-0.32, 0.32):
                    pts.append((float(base[0] + dx), float(base[2])))
    return pts


def check(
    states: list[PoseState],
    choreo=None,
    analysis=None,
    fps: float = FPS,
) -> dict:
    metrics: list[Metric] = []
    n = len(states)

    # ---------------- foot slide ----------------------------------------
    slides = []
    per_side = {}
    for side in ("L", "R"):
        prev = None
        side_slides = []
        for st in states:
            ankle = st.pos[f"ankle.{side}"]
            toe = st.pos[f"toe.{side}"]
            contact = toe[1] < 0.80 and ankle[1] < 1.25
            if contact and prev is not None and prev[2]:
                d = math.hypot(ankle[0] - prev[0], ankle[2] - prev[1])
                side_slides.append(d)
            prev = (float(ankle[0]), float(ankle[2]), contact)
        per_side[side] = side_slides
        slides += side_slides
    if slides:
        arr = np.asarray(slides)
        cm_per_s = arr * UNIT_CM * fps
        slide_mean = float(np.mean(cm_per_s))
        slide_p95 = float(np.percentile(cm_per_s, 95))
    else:
        slide_mean = slide_p95 = 0.0
    metrics.append(
        Metric(
            "foot_slide",
            slide_p95,
            "cm/s (p95)",
            _status(slide_p95, 3.0, 8.0),
            f"mean {slide_mean:.2f} cm/s over {len(slides)} contact frames",
        )
    )

    # ---------------- penetration ---------------------------------------
    min_foot = min(
        min(min(st.pos[f"toe.{s}"][1] - 0.30, st.pos[f"ankle.{s}"][1] - 0.85) for s in ("L", "R"))
        for st in states
    )
    penetration_cm = max(0.0, -min_foot) * UNIT_CM
    metrics.append(
        Metric("ground_penetration", penetration_cm, "cm", _status(penetration_cm, 0.8, 2.5))
    )

    # ---------------- balance -------------------------------------------
    margins = []
    for st in states:
        pts = contact_points(st)
        if not pts:
            continue
        com = centre_of_mass(st)
        margins.append(_point_margin((float(com[0]), float(com[2])), _hull(pts)))
    if margins:
        m = np.asarray(margins) * UNIT_CM
        outside_ratio = float(np.mean(m < 0)) * 100.0
        metrics.append(
            Metric(
                "balance_outside",
                outside_ratio,
                "% frames",
                _status(outside_ratio, 22.0, 45.0),
                f"mean margin {float(np.mean(m)):.1f} cm",
            )
        )
    else:
        metrics.append(Metric("balance_outside", 0.0, "% frames", "PASS", "no contact frames"))

    # ---------------- joint limits --------------------------------------
    violations = 0
    worst = ("", 0.0)
    if choreo is not None and analysis is not None:
        from .rig import LIMITS, DEFAULT_LIMIT

        for st in states:
            pose, _ = compose_pose(choreo, analysis, st.beat, st.energy)
            for joint, (rx, ry, rz) in pose.items():
                for axis, v in enumerate((rx, ry, rz)):
                    lo, hi = LIMITS.get((joint, axis), DEFAULT_LIMIT)
                    over = max(0.0, v - hi, lo - v)
                    if over > 0.5:
                        violations += 1
                        if over > worst[1]:
                            worst = (f"{joint}[{axis}]", over)
    metrics.append(
        Metric(
            "joint_limit_hits",
            float(violations),
            "frames",
            _status(float(violations), 1.0, 60.0),
            f"worst {worst[0]} by {worst[1]:.1f} deg" if worst[0] else "",
        )
    )

    # ---------------- beat alignment ------------------------------------
    motion = []
    for a, b in zip(states, states[1:]):
        e = 0.0
        for bone, qa in a.local.items():
            qb = b.local.get(bone)
            if qb is None:
                continue
            e += quat_angle_between(qa, qb)
        motion.append(e)
    motion = np.asarray(motion)

    # ---------------- beat alignment ------------------------------------
    # The body drops onto the beat, so the minima of the root height are the
    # musical accents this pipeline is supposed to place on the grid.
    hit_ratio = 0.0
    mean_offset_ms = 0.0
    if analysis is not None and len(states) > 4:
        height = np.asarray([float(st.pos["root"][1]) for st in states])
        minima = [
            i
            for i in range(1, len(height) - 1)
            if height[i] < height[i - 1] - 1e-5 and height[i] <= height[i + 1] and height[i] < np.median(height)
        ]
        if minima:
            beat_times = analysis.beat_times
            offs = np.asarray([float(np.min(np.abs(beat_times - i / fps))) for i in minima])
            hit_ratio = float(np.mean(offs < 0.07)) * 100.0
            mean_offset_ms = float(np.mean(offs)) * 1000.0
    metrics.append(
        Metric(
            "beat_alignment",
            hit_ratio,
            "% accents on beat",
            "PASS" if hit_ratio >= 55 else ("WARN" if hit_ratio >= 35 else "FAIL"),
            f"mean offset {mean_offset_ms:.0f} ms",
        )
    )

    # ---------------- smoothness ----------------------------------------
    # p90 of the per frame worst joint acceleration: intentional hits are rare
    # and must not dominate a "does the whole clip jitter" check.
    snap = 0.0
    if len(states) > 3:
        per_frame = []
        for i in range(1, len(states) - 1):
            worst = 0.0
            for bone in states[i].local:
                d1 = quat_angle_between(states[i - 1].local[bone], states[i].local[bone])
                d2 = quat_angle_between(states[i].local[bone], states[i + 1].local[bone])
                worst = max(worst, abs(d2 - d1))
            per_frame.append(worst)
        snap = float(np.percentile(per_frame, 90))
    metrics.append(
        Metric("snap", snap, "deg/frame^2 (p90)", _status(snap, 6.0, 10.0))
    )

    overall = "PASS"
    if any(m.status == "FAIL" for m in metrics):
        overall = "FAIL"
    elif any(m.status == "WARN" for m in metrics):
        overall = "WARN"

    return {
        "frames": n,
        "duration_s": clean(n / fps),
        "overall": overall,
        "metrics": [m.to_dict() for m in metrics],
        "foot_slide_detail": {
            side: clean(float(np.mean(np.asarray(v) * UNIT_CM * fps))) if v else 0.0
            for side, v in per_side.items()
        },
    }


def clean(v: float) -> float:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return round(float(v), 3)


def check_vmd(motion, expected_bones: set[str] | None = None) -> dict:
    """Structural checks on the baked file itself (reads what MMD will read)."""
    from .rig import JOINT_LIST

    allowed = expected_bones or {j.mmd for j in JOINT_LIST}
    names = motion.bone_names()
    unknown = sorted(set(names) - allowed)

    seen = set()
    dup = 0
    bad_quat = 0
    for k in motion.bones:
        key = (k.bone, k.frame)
        if key in seen:
            dup += 1
        seen.add(key)
        q = np.asarray(k.rotation)
        if abs(float(np.linalg.norm(q)) - 1.0) > 1e-3:
            bad_quat += 1

    frames = [k.frame for k in motion.bones]
    out = {
        "bones": len(names),
        "keys": len(motion.bones),
        "frame_range": [min(frames), max(frames)] if frames else [0, 0],
        "unknown_bones": unknown,
        "duplicate_keys": dup,
        "non_unit_quaternions": bad_quat,
        "status": "PASS" if (not unknown and dup == 0 and bad_quat == 0) else "FAIL",
    }
    return out


def format_report(report: dict) -> str:
    lines = [f"L4 verification  overall={report['overall']}  frames={report['frames']}"]
    for m in report["metrics"]:
        detail = f"  ({m['detail']})" if m["detail"] else ""
        lines.append(f"  [{m['status']:4s}] {m['name']:<20} {m['value']:>10.3f} {m['unit']}{detail}")
    return "\n".join(lines)
