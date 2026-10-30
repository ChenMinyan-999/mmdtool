"""Canonical humanoid rig + MMD bone name mapping + pose helpers.

Canonical space: Y up, character faces +Z, character's LEFT is +X (right-handed).
Units are MMD units (1 unit ~= 8 cm, so the default rig is ~19.5 units ~ 156 cm).

The rig is intentionally *rigid*: every bone has an identity rest orientation,
exactly like a plain MMD model, so a VMD local rotation can be authored directly
in this space without any rest-orientation bookkeeping.

Pose helpers (``arm``, ``leg``, ``spine`` ...) speak in semantics ("arm up 120
degrees") instead of Euler numbers, which is the whole point of the intermediate
layer: an LLM (or a human) writes semantics, the compiler writes numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

UNIT_CM = 8.0  # 1 MMD unit in centimetres (approximate, used for reporting only)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    rest: tuple[float, float, float]
    mmd: str
    side: int = 0  # +1 = character left, -1 = character right, 0 = centre
    channel: str = "torso"


# --------------------------------------------------------------------------- #
# skeleton
# --------------------------------------------------------------------------- #
# fmt: off
JOINT_LIST = [
    Joint("root",       None,       (0.00, 10.00, 0.00), "センター",   0, "root"),
    Joint("lower",      "root",     (0.00, 10.00, 0.00), "下半身",     0, "torso"),
    Joint("upper",      "lower",    (0.00, 10.35, 0.00), "上半身",     0, "torso"),
    Joint("chest",      "upper",    (0.00, 12.60, 0.00), "上半身2",    0, "torso"),
    Joint("neck",       "chest",    (0.00, 14.60, 0.00), "首",         0, "head"),
    Joint("head",       "neck",     (0.00, 15.40, 0.00), "頭",         0, "head"),

    Joint("shoulder.L", "chest",    (0.35, 14.35, 0.00), "左肩",      +1, "arm_l"),
    Joint("arm.L",      "shoulder.L", (1.40, 14.20, 0.00), "左腕",    +1, "arm_l"),
    Joint("elbow.L",    "arm.L",    (4.20, 14.20, 0.00), "左ひじ",    +1, "arm_l"),
    Joint("wrist.L",    "elbow.L",  (6.90, 14.20, 0.00), "左手首",    +1, "arm_l"),

    Joint("shoulder.R", "chest",    (-0.35, 14.35, 0.00), "右肩",     -1, "arm_r"),
    Joint("arm.R",      "shoulder.R", (-1.40, 14.20, 0.00), "右腕",   -1, "arm_r"),
    Joint("elbow.R",    "arm.R",    (-4.20, 14.20, 0.00), "右ひじ",   -1, "arm_r"),
    Joint("wrist.R",    "elbow.R",  (-6.90, 14.20, 0.00), "右手首",   -1, "arm_r"),

    Joint("leg.L",      "lower",    (0.95, 10.00, 0.00), "左足",      +1, "legs"),
    Joint("knee.L",     "leg.L",    (0.98, 5.60, 0.00), "左ひざ",     +1, "legs"),
    Joint("ankle.L",    "knee.L",   (0.98, 0.90, 0.00), "左足首",     +1, "legs"),
    Joint("toe.L",      "ankle.L",  (0.98, 0.35, 1.35), "左つま先",   +1, "legs"),

    Joint("leg.R",      "lower",    (-0.95, 10.00, 0.00), "右足",     -1, "legs"),
    Joint("knee.R",     "leg.R",    (-0.98, 5.60, 0.00), "右ひざ",    -1, "legs"),
    Joint("ankle.R",    "knee.R",   (-0.98, 0.90, 0.00), "右足首",    -1, "legs"),
    Joint("toe.R",      "ankle.R",  (-0.98, 0.35, 1.35), "右つま先",  -1, "legs"),

    # IK bones are *not* part of the deform skeleton; they are emitted so that
    # models with leg-IK enabled reproduce the exact same pose.
    Joint("leg_ik.L",   "root",     (0.98, 0.90, 0.00), "左足ＩＫ",   +1, "ik"),
    Joint("toe_ik.L",   "leg_ik.L", (0.98, 0.35, 1.35), "左つま先ＩＫ", +1, "ik"),
    Joint("leg_ik.R",   "root",     (-0.98, 0.90, 0.00), "右足ＩＫ",  -1, "ik"),
    Joint("toe_ik.R",   "leg_ik.R", (-0.98, 0.35, 1.35), "右つま先ＩＫ", -1, "ik"),
]
# fmt: on

JOINTS: dict[str, Joint] = {j.name: j for j in JOINT_LIST}

#: bones actually driven by the animator (IK bones are derived)
DEFORM_BONES = [j.name for j in JOINT_LIST if j.channel != "ik"]

#: stable output order for VMD
BONE_ORDER = [j.name for j in JOINT_LIST]

CHANNELS = ["root", "torso", "head", "arm_l", "arm_r", "legs"]

CHANNEL_BONES = {c: [j.name for j in JOINT_LIST if j.channel == c] for c in CHANNELS}


def rest_position(name: str):
    return JOINTS[name].rest


def parent_of(name: str):
    return JOINTS[name].parent


# --------------------------------------------------------------------------- #
# semantic pose helpers  (all values in degrees, all returns (rx, ry, rz))
# --------------------------------------------------------------------------- #
ARM_DOWN = 78.0  # rz offset that takes a T-pose arm to "hanging down"


def arm(side: int, up: float = 0.0, fwd: float = 0.0, twist: float = 0.0):
    """Arm pose. ``up`` 0 = hanging down, 78 = T-pose, 170 = overhead."""
    return (twist * side, -fwd * side, (up - ARM_DOWN) * side)


def elbow(side: int, bend: float = 0.0, spread: float = 0.0):
    """Elbow flexion: 0 = straight, 90 = forearm forward. ``spread`` opens it up."""
    return (0.0, -bend * side, spread * side)


def wrist(side: int, bend: float = 0.0, flick: float = 0.0):
    return (flick * side, -bend * side, 0.0)


def shoulder(side: int, up: float = 0.0, fwd: float = 0.0):
    return (0.0, -fwd * side, up * side)


def leg(side: int, fwd: float = 0.0, out: float = 0.0, twist: float = 0.0):
    """Leg swing from vertical. ``fwd`` positive = knee travels forward."""
    return (-fwd, twist * side, out * side)


def knee(bend: float = 0.0):
    """Knee flexion: 0 = straight, positive = heel towards the butt."""
    return (bend, 0.0, 0.0)


def ankle(pitch: float = 0.0, roll: float = 0.0):
    """``pitch`` positive = toes point down."""
    return (pitch, 0.0, roll)


def spine(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
    """Torso. pitch + = lean forward, yaw + = turn to character's left,
    roll + = lean towards character's right."""
    return (pitch, yaw, roll)


def head(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
    return (pitch, yaw, roll)


# --------------------------------------------------------------------------- #
# base pose: relaxed standing, arms down. Every authored pose is absolute.
# --------------------------------------------------------------------------- #
def _base_pose() -> dict[str, tuple[float, float, float]]:
    p: dict[str, tuple[float, float, float]] = {name: (0.0, 0.0, 0.0) for name in DEFORM_BONES}
    for side, s in (("L", +1), ("R", -1)):
        p[f"shoulder.{side}"] = shoulder(s, up=-4.0)
        p[f"arm.{side}"] = arm(s, up=4.0, fwd=2.0, twist=-6.0)
        p[f"elbow.{side}"] = elbow(s, bend=10.0, spread=3.0)
        p[f"wrist.{side}"] = wrist(s, bend=4.0)
        p[f"leg.{side}"] = leg(s, fwd=0.0, out=0.5)
        p[f"knee.{side}"] = knee(3.0)
        p[f"ankle.{side}"] = ankle(pitch=-2.0)
        p[f"toe.{side}"] = (0.0, 0.0, 0.0)
    p["lower"] = spine(pitch=-2.0)
    p["upper"] = spine(pitch=2.0)
    p["chest"] = spine(pitch=-1.0)
    p["head"] = head(pitch=1.0)
    return p


BASE_POSE = _base_pose()

#: soft joint limits (degrees) used by the validator; keys are (joint, axis 0..2).
#: Limits are authored for the left side and mirrored with (rx, -ry, -rz).
_LEFT_LIMITS: dict[tuple[str, int], tuple[float, float]] = {
    ("arm.L", 2): (-100.0, 100.0),   # rz (raise/lower)
    ("arm.L", 1): (-175.0, 60.0),    # ry (forward swing)
    ("elbow.L", 1): (-155.0, 5.0),   # ry (flexion)
    ("leg.L", 0): (-125.0, 45.0),    # rx (hip flexion)
    ("knee.L", 0): (-3.0, 150.0),    # rx
    ("ankle.L", 0): (-45.0, 55.0),
    ("shoulder.L", 2): (-40.0, 40.0),
    ("shoulder.L", 1): (-25.0, 25.0),
    ("toe.L", 0): (-30.0, 60.0),
}

LIMITS: dict[tuple[str, int], tuple[float, float]] = dict(_LEFT_LIMITS)
for _key, (_lo, _hi) in _LEFT_LIMITS.items():
    _name, _axis = _key
    if _name.endswith(".L"):
        LIMITS[(_name[:-2] + ".R", _axis)] = (-_hi, -_lo) if _axis in (1, 2) else (_lo, _hi)
del _key, _lo, _hi, _name, _axis

LIMITS[("upper", 0)] = (-30.0, 40.0)
LIMITS[("upper", 1)] = (-50.0, 50.0)
LIMITS[("upper", 2)] = (-30.0, 30.0)
LIMITS[("chest", 0)] = (-25.0, 30.0)
LIMITS[("chest", 1)] = (-45.0, 45.0)
LIMITS[("chest", 2)] = (-28.0, 28.0)
LIMITS[("lower", 0)] = (-20.0, 25.0)
LIMITS[("lower", 1)] = (-35.0, 35.0)
LIMITS[("lower", 2)] = (-25.0, 25.0)
LIMITS[("neck", 0)] = (-35.0, 35.0)
LIMITS[("neck", 1)] = (-55.0, 55.0)
LIMITS[("neck", 2)] = (-25.0, 25.0)
LIMITS[("head", 0)] = (-25.0, 25.0)
LIMITS[("head", 1)] = (-40.0, 40.0)
LIMITS[("head", 2)] = (-25.0, 25.0)

DEFAULT_LIMIT = (-180.0, 180.0)


def clamp_pose(pose: dict[str, tuple[float, float, float]]):
    out = {}
    for name, (rx, ry, rz) in pose.items():
        out[name] = tuple(
            min(max(v, LIMITS.get((name, i), DEFAULT_LIMIT)[0]), LIMITS.get((name, i), DEFAULT_LIMIT)[1])
            for i, v in enumerate((rx, ry, rz))
        )
    return out


def mirror_name(name: str) -> str:
    if name.endswith(".L"):
        return name[:-2] + ".R"
    if name.endswith(".R"):
        return name[:-2] + ".L"
    return name
