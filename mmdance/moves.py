"""Motion primitive library ("moves").

A move is a short, human-authored phrase described by *semantic* poses, in the
spirit of MPL::

    ("arm.R", arm(-1, up=150, fwd=15))      # right arm points up and forwards
    ("knee.L", knee(70))                    # left knee bent 70 degrees

Keyframes are ``(phase, pose)`` with ``phase`` in 0..1 over the move length.
Only the joints a key mentions are overridden; everything else falls back to
:data:`mmdance.rig.BASE_POSE`, which keeps authoring compact.

An LLM can produce new entries in exactly this shape (see ``mmdance.choreo``),
and the validator keeps it honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .mathx import lerp, smoothstep
from .rig import BASE_POSE, arm, ankle, elbow, head, knee, leg, shoulder, spine

Pose = dict[str, tuple[float, float, float]]


def _full(pose: Pose) -> Pose:
    out = dict(BASE_POSE)
    out.update(pose)
    return out


@dataclass
class Move:
    name: str
    beats: float
    channels: tuple[str, ...]
    energy: tuple[float, float]  # usable energy band
    tags: tuple[str, ...] = ()
    keys: list[tuple[float, Pose]] = field(default_factory=list)
    root_keys: list[tuple[float, tuple[float, float, float]]] = field(default_factory=list)
    description: str = ""

    def _prepared(self):
        if not hasattr(self, "_keys"):
            self._keys = [(p, _full(dict(q))) for p, q in self.keys]
            self._roots = sorted(self.root_keys, key=lambda k: k[0])
        return self._keys, self._roots

    # ------------------------------------------------------------------ #
    def pose_at(self, phase: float) -> Pose:
        """Interpolated absolute pose at ``phase`` in 0..1."""
        keys, _ = self._prepared()
        phase = min(1.0, max(0.0, phase))
        if not keys:
            return dict(BASE_POSE)
        if phase <= keys[0][0]:
            return dict(keys[0][1])
        for (p0, a), (p1, b) in zip(keys, keys[1:]):
            if p0 <= phase <= p1:
                t = smoothstep((phase - p0) / max(1e-9, p1 - p0))
                return {
                    j: tuple(lerp(a[j][i], b[j][i], t) for i in range(3))  # type: ignore[misc]
                    for j in a
                }
        return dict(keys[-1][1])

    def root_at(self, phase: float) -> tuple[float, float, float]:
        _, roots = self._prepared()
        phase = min(1.0, max(0.0, phase))
        if not roots:
            return (0.0, 0.0, 0.0)
        if phase <= roots[0][0]:
            return roots[0][1]
        for (p0, a), (p1, b) in zip(roots, roots[1:]):
            if p0 <= phase <= p1:
                t = smoothstep((phase - p0) / max(1e-9, p1 - p0))
                return tuple(lerp(a[i], b[i], t) for i in range(3))  # type: ignore[return-value]
        return roots[-1][1]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "beats": self.beats,
            "channels": list(self.channels),
            "energy": list(self.energy),
            "tags": list(self.tags),
            "description": self.description,
        }


# --------------------------------------------------------------------------- #
# the library
# --------------------------------------------------------------------------- #
LIBRARY: dict[str, Move] = {}


def move(name, beats, channels, energy, tags=(), description=""):
    def deco(fn):
        keys, roots = fn()
        LIBRARY[name] = Move(
            name=name,
            beats=beats,
            channels=tuple(channels),
            energy=tuple(energy),
            tags=tuple(tags),
            keys=keys,
            root_keys=roots,
            description=description,
        )
        return fn

    return deco


def S(side: str) -> int:
    return 1 if side == "L" else -1


# --- calm / low energy -------------------------------------------------------
@move("sway", beats=4, channels=("torso", "head", "arm_l", "arm_r"), energy=(0.0, 0.75),
      tags=("groove", "calm"), description="Weight shift left/right, relaxed arms.")
def _sway():
    def k(roll, yaw, arm_l, arm_r, head_roll):
        return {
            "lower": spine(roll=roll * 0.5, yaw=-yaw),
            "upper": spine(roll=roll, yaw=yaw, pitch=1.5),
            "chest": spine(roll=roll * 0.6, yaw=yaw * 0.5),
            "head": head(roll=head_roll, yaw=-yaw * 0.6),
            "arm.L": arm(+1, up=6, fwd=6),
            "arm.R": arm(-1, up=6, fwd=6),
            "elbow.L": elbow(+1, bend=18 + arm_l),
            "elbow.R": elbow(-1, bend=18 + arm_r),
        }

    return [
        (0.00, k(0, 0, 0, 0, 0)),
        (0.25, k(-5, -6, 4, 10, 3)),
        (0.50, k(0, 0, 0, 0, 0)),
        (0.75, k(5, 6, 10, 4, -3)),
        (1.00, k(0, 0, 0, 0, 0)),
    ], []


@move("body_roll", beats=4, channels=("torso", "head", "arm_l", "arm_r"), energy=(0.2, 1.0),
      tags=("groove", "expressive"), description="Slow chest roll, arms floating.")
def _body_roll():
    def k(pitch, roll, arm_up, head_p):
        return {
            "lower": spine(pitch=-pitch * 0.4, roll=roll * 0.3),
            "upper": spine(pitch=pitch * 0.7, roll=roll),
            "chest": spine(pitch=pitch, roll=roll * 1.2),
            "neck": head(pitch=-pitch * 0.4),
            "head": head(pitch=head_p, roll=-roll * 0.5),
            "arm.L": arm(+1, up=arm_up, fwd=12),
            "arm.R": arm(-1, up=arm_up, fwd=12),
            "elbow.L": elbow(+1, bend=36),
            "elbow.R": elbow(-1, bend=36),
        }

    return [
        (0.00, k(0, 0, 8, 0)),
        (0.25, k(-6, -7, 26, -4)),
        (0.50, k(4, 0, 8, 3)),
        (0.75, k(-6, 7, 26, -4)),
        (1.00, k(0, 0, 8, 0)),
    ], []


@move("freeze", beats=4, channels=("torso", "head", "arm_l", "arm_r"), energy=(0.0, 0.5),
      tags=("calm", "break"), description="Hold a diagonal pose - breathing room.")
def _freeze():
    pose = {
        "upper": spine(pitch=-3, yaw=-8, roll=4),
        "chest": spine(pitch=-4, yaw=-6, roll=3),
        "head": head(yaw=-12, roll=-5, pitch=-3),
        "arm.L": arm(+1, up=28, fwd=18),
        "arm.R": arm(-1, up=52, fwd=-10),
        "elbow.L": elbow(+1, bend=48),
        "elbow.R": elbow(-1, bend=62),
        "leg.L": leg(+1, fwd=4, out=3),
        "leg.R": leg(-1, fwd=-2, out=2),
        "knee.L": knee(12),
        "knee.R": knee(8),
    }
    return [(0.0, pose), (0.85, pose), (1.0, pose)], []


# --- medium ----------------------------------------------------------------
@move("arm_pump_r", beats=2, channels=("arm_r", "torso", "head"), energy=(0.45, 1.0),
      tags=("accent", "arms"), description="Right arm pumps up on the beat.")
def _arm_pump_r():
    return [
        (0.00, {"arm.R": arm(-1, up=8, fwd=4), "elbow.R": elbow(-1, bend=24)}),
        (0.22, {
            "arm.R": arm(-1, up=132, fwd=18),
            "elbow.R": elbow(-1, bend=18),
            "upper": spine(roll=-4),
            "chest": spine(roll=-3, yaw=-6),
            "head": head(roll=-4, pitch=-4),
        }),
        (0.60, {
            "arm.R": arm(-1, up=132, fwd=18),
            "elbow.R": elbow(-1, bend=18),
            "upper": spine(roll=-4),
            "chest": spine(roll=-3, yaw=-6),
            "head": head(roll=-4, pitch=-4),
        }),
        (1.00, {"arm.R": arm(-1, up=8, fwd=4), "elbow.R": elbow(-1, bend=24)}),
    ], []


@move("arm_pump_l", beats=2, channels=("arm_l", "torso", "head"), energy=(0.45, 1.0),
      tags=("accent", "arms"), description="Left arm pumps up on the beat.")
def _arm_pump_l():
    return [
        (0.00, {"arm.L": arm(+1, up=8, fwd=4), "elbow.L": elbow(+1, bend=24)}),
        (0.22, {
            "arm.L": arm(+1, up=132, fwd=18),
            "elbow.L": elbow(+1, bend=18),
            "upper": spine(roll=4),
            "chest": spine(roll=3, yaw=6),
            "head": head(roll=4, pitch=-4),
        }),
        (0.60, {
            "arm.L": arm(+1, up=132, fwd=18),
            "elbow.L": elbow(+1, bend=18),
            "upper": spine(roll=4),
            "chest": spine(roll=3, yaw=6),
            "head": head(roll=4, pitch=-4),
        }),
        (1.00, {"arm.L": arm(+1, up=8, fwd=4), "elbow.L": elbow(+1, bend=24)}),
    ], []


@move("point_side_r", beats=2, channels=("arm_r", "torso", "head"), energy=(0.5, 1.0),
      tags=("accent", "arms"), description="Point to the right, chin up.")
def _point_side_r():
    pose = {
        "arm.R": arm(-1, up=76, fwd=6),
        "elbow.R": elbow(-1, bend=2),
        "chest": spine(yaw=-10, roll=-3),
        "upper": spine(yaw=-6),
        "head": head(yaw=-16),
        "leg.R": leg(-1, out=6),
        "knee.R": knee(6),
    }
    return [(0.0, {}), (0.25, pose), (0.70, pose), (1.0, {})], []


@move("clap", beats=2, channels=("arm_l", "arm_r", "torso"), energy=(0.55, 1.0),
      tags=("accent", "arms"), description="Clap in front on the beat.")
def _clap():
    hit = {
        "arm.L": arm(+1, up=44, fwd=62),
        "arm.R": arm(-1, up=44, fwd=62),
        "elbow.L": elbow(+1, bend=74),
        "elbow.R": elbow(-1, bend=74),
        "chest": spine(pitch=3),
        "head": head(pitch=2),
        "knee.L": knee(10),
        "knee.R": knee(10),
    }
    open_ = {
        "arm.L": arm(+1, up=36, fwd=40),
        "arm.R": arm(-1, up=36, fwd=40),
        "elbow.L": elbow(+1, bend=58),
        "elbow.R": elbow(-1, bend=58),
    }
    return [(0.0, open_), (0.12, hit), (0.35, hit), (1.0, open_)], []


@move("hip_bump_r", beats=2, channels=("torso", "legs", "arm_l", "arm_r"), energy=(0.5, 1.0),
      tags=("accent", "hips"), description="Hip push to the right with arm counter.")
def _hip_bump_r():
    hit = {
        "lower": spine(roll=-11, yaw=6),
        "upper": spine(roll=7, yaw=-5),
        "chest": spine(roll=4),
        "head": head(roll=5, yaw=-6),
        "leg.L": leg(+1, out=9),
        "knee.L": knee(16),
        "leg.R": leg(-1, out=-3),
        "knee.R": knee(4),
        "arm.L": arm(+1, up=34, fwd=10),
        "arm.R": arm(-1, up=18, fwd=-14),
        "elbow.L": elbow(+1, bend=52),
        "elbow.R": elbow(-1, bend=34),
    }
    return [(0.0, {}), (0.20, hit), (0.45, hit), (1.0, {})], [((0.2, (-0.35, 0.0, 0.0))), (0.45, (-0.35, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))]


@move("step_touch_r", beats=4, channels=("legs", "torso", "arm_l", "arm_r"), energy=(0.3, 1.0),
      tags=("travel", "legs"), description="Step to the right, touch, shift back.")
def _step_touch_r():
    out = {
        "leg.R": leg(-1, out=16, fwd=2),
        "knee.R": knee(14),
        "leg.L": leg(+1, out=-2),
        "lower": spine(roll=-6),
        "upper": spine(roll=4, yaw=-4),
        "head": head(roll=3, yaw=-6),
        "arm.L": arm(+1, up=22, fwd=8),
        "arm.R": arm(-1, up=10, fwd=-6),
        "elbow.L": elbow(+1, bend=42),
        "elbow.R": elbow(-1, bend=26),
    }
    touch = {
        "leg.R": leg(-1, out=6, fwd=6),
        "knee.R": knee(34),
        "ankle.R": ankle(pitch=16),
        "leg.L": leg(+1, out=2),
        "knee.L": knee(8),
        "lower": spine(roll=-3),
        "upper": spine(roll=2),
        "arm.L": arm(+1, up=14, fwd=4),
        "arm.R": arm(-1, up=8, fwd=0),
    }
    return [
        (0.00, {}),
        (0.25, out),
        (0.50, touch),
        (0.75, out),
        (1.00, {}),
    ], [(0.0, (0.0, 0.0, 0.0)), (0.25, (-0.55, 0.0, 0.0)), (0.5, (-0.9, 0.0, 0.0)), (0.75, (-0.55, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))]


@move("step_touch_l", beats=4, channels=("legs", "torso", "arm_l", "arm_r"), energy=(0.3, 1.0),
      tags=("travel", "legs"), description="Step to the left, touch, shift back.")
def _step_touch_l():
    out = {
        "leg.L": leg(+1, out=16, fwd=2),
        "knee.L": knee(14),
        "leg.R": leg(-1, out=-2),
        "lower": spine(roll=6),
        "upper": spine(roll=-4, yaw=4),
        "head": head(roll=-3, yaw=6),
        "arm.R": arm(-1, up=22, fwd=8),
        "arm.L": arm(+1, up=10, fwd=-6),
        "elbow.R": elbow(-1, bend=42),
        "elbow.L": elbow(+1, bend=26),
    }
    touch = {
        "leg.L": leg(+1, out=6, fwd=6),
        "knee.L": knee(34),
        "ankle.L": ankle(pitch=16),
        "leg.R": leg(-1, out=2),
        "knee.R": knee(8),
        "lower": spine(roll=3),
        "upper": spine(roll=-2),
        "arm.R": arm(-1, up=14, fwd=4),
        "arm.L": arm(+1, up=8, fwd=0),
    }
    return [
        (0.00, {}),
        (0.25, out),
        (0.50, touch),
        (0.75, out),
        (1.00, {}),
    ], [(0.0, (0.0, 0.0, 0.0)), (0.25, (0.55, 0.0, 0.0)), (0.5, (0.9, 0.0, 0.0)), (0.75, (0.55, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))]


# --- high energy ------------------------------------------------------------
@move("arms_wide", beats=4, channels=("arm_l", "arm_r", "torso", "head"), energy=(0.55, 1.0),
      tags=("accent", "arms", "big"), description="Open both arms wide, chest up.")
def _arms_wide():
    wide = {
        "arm.L": arm(+1, up=88, fwd=10),
        "arm.R": arm(-1, up=88, fwd=10),
        "elbow.L": elbow(+1, bend=8),
        "elbow.R": elbow(-1, bend=8),
        "upper": spine(pitch=-5),
        "chest": spine(pitch=-3),
        "head": head(pitch=-6),
        "shoulder.L": shoulder(+1, up=6),
        "shoulder.R": shoulder(-1, up=6),
    }
    return [(0.0, {}), (0.30, wide), (0.68, wide), (1.0, {})], []


@move("point_up_r", beats=4, channels=("arm_r", "torso", "head"), energy=(0.6, 1.0),
      tags=("accent", "arms", "big"), description="Right arm points to the sky.")
def _point_up_r():
    up = {
        "arm.R": arm(-1, up=158, fwd=14),
        "elbow.R": elbow(-1, bend=4),
        "shoulder.R": shoulder(-1, up=10),
        "chest": spine(yaw=-8, roll=-6, pitch=-4),
        "upper": spine(yaw=-5, roll=-3),
        "neck": head(yaw=-6, pitch=-8),
        "head": head(yaw=-8, pitch=-6),
        "leg.L": leg(+1, out=5),
        "knee.L": knee(10),
    }
    return [(0.0, {}), (0.22, up), (0.62, up), (1.0, {})], []


@move("wave_arms", beats=4, channels=("arm_l", "arm_r", "torso", "head"), energy=(0.6, 1.0),
      tags=("accent", "arms", "big"), description="Both arms overhead, waving side to side.")
def _wave_arms():
    def k(l_up, r_up, roll):
        return {
            "arm.L": arm(+1, up=l_up, fwd=8),
            "arm.R": arm(-1, up=r_up, fwd=8),
            "elbow.L": elbow(+1, bend=10),
            "elbow.R": elbow(-1, bend=10),
            "upper": spine(roll=roll),
            "chest": spine(roll=roll * 0.7),
            "head": head(roll=-roll * 0.6),
        }

    return [
        (0.00, k(120, 120, 0)),
        (0.25, k(150, 96, -6)),
        (0.50, k(120, 120, 0)),
        (0.75, k(96, 150, 6)),
        (1.00, k(120, 120, 0)),
    ], []


@move("kick_l", beats=2, channels=("legs", "torso", "arm_l", "arm_r"), energy=(0.6, 1.0),
      tags=("accent", "legs", "big"), description="Left front kick.")
def _kick_l():
    hit = {
        "leg.L": leg(+1, fwd=58),
        "knee.L": knee(8),
        "ankle.L": ankle(pitch=22),
        "leg.R": leg(-1, fwd=-6),
        "knee.R": knee(16),
        "lower": spine(roll=4),
        "upper": spine(roll=-6, pitch=-4),
        "head": head(pitch=-4),
        "arm.L": arm(+1, up=30, fwd=-24),
        "arm.R": arm(-1, up=24, fwd=30),
        "elbow.L": elbow(+1, bend=46),
        "elbow.R": elbow(-1, bend=40),
    }
    return [(0.0, {}), (0.18, hit), (0.42, hit), (1.0, {})], []


@move("kick_r", beats=2, channels=("legs", "torso", "arm_l", "arm_r"), energy=(0.6, 1.0),
      tags=("accent", "legs", "big"), description="Right front kick.")
def _kick_r():
    hit = {
        "leg.R": leg(-1, fwd=58),
        "knee.R": knee(8),
        "ankle.R": ankle(pitch=22),
        "leg.L": leg(+1, fwd=-6),
        "knee.L": knee(16),
        "lower": spine(roll=-4),
        "upper": spine(roll=6, pitch=-4),
        "head": head(pitch=-4),
        "arm.R": arm(-1, up=30, fwd=-24),
        "arm.L": arm(+1, up=24, fwd=30),
        "elbow.R": elbow(-1, bend=46),
        "elbow.L": elbow(+1, bend=40),
    }
    return [(0.0, {}), (0.18, hit), (0.42, hit), (1.0, {})], []


@move("pump_down", beats=2, channels=("arm_l", "arm_r", "torso", "legs"), energy=(0.6, 1.0),
      tags=("accent", "arms"), description="Both hands push down on the beat, knees dip.")
def _pump_down():
    hit = {
        "arm.L": arm(+1, up=4, fwd=26),
        "arm.R": arm(-1, up=4, fwd=26),
        "elbow.L": elbow(+1, bend=14),
        "elbow.R": elbow(-1, bend=14),
        "upper": spine(pitch=8),
        "chest": spine(pitch=6),
        "head": head(pitch=8),
        "knee.L": knee(26),
        "knee.R": knee(26),
        "leg.L": leg(+1, fwd=8),
        "leg.R": leg(-1, fwd=8),
    }
    return [(0.0, {}), (0.15, hit), (0.38, hit), (1.0, {})], [((0.15, (0.0, -0.45, 0.0))), (0.38, (0.0, -0.45, 0.0)), (1.0, (0.0, 0.0, 0.0))]


@move("turn_pose", beats=4, channels=("torso", "head", "arm_l", "arm_r", "legs"), energy=(0.5, 1.0),
      tags=("accent", "torso"), description="Quarter turn with a pointed arm, then back.")
def _turn_pose():
    p = {
        "lower": spine(yaw=-14),
        "upper": spine(yaw=26, roll=5),
        "chest": spine(yaw=18, roll=4),
        "head": head(yaw=24, roll=-4),
        "arm.L": arm(+1, up=58, fwd=-30),
        "arm.R": arm(-1, up=30, fwd=52),
        "elbow.L": elbow(+1, bend=26),
        "elbow.R": elbow(-1, bend=64),
        "leg.L": leg(+1, out=6, twist=10),
        "leg.R": leg(-1, out=-4, twist=-6),
        "knee.L": knee(16),
        "knee.R": knee(6),
    }
    return [(0.0, {}), (0.35, p), (0.65, p), (1.0, {})], []


def get(name: str) -> Move:
    if name not in LIBRARY:
        raise KeyError(f"unknown move {name!r}; known: {sorted(LIBRARY)}")
    return LIBRARY[name]


def catalog() -> list[dict]:
    return [m.as_dict() for m in sorted(LIBRARY.values(), key=lambda m: m.name)]


def export_catalog(path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"version": "1.0", "moves": catalog()}, fh, ensure_ascii=False, indent=2)
