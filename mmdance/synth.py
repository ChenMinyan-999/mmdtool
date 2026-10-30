"""L2 - choreography score + groove -> per-frame skeletal pose.

Three layers are added in this order:

1. ``BASE_POSE``         relaxed standing pose (rig.py)
2. move deltas           weighted blend of the score's active slots
3. groove                beat-locked procedural layer (bounce, sway, arm swing)

then forward kinematics runs, feet are locked against sliding with a two-bone
IK pass, and the result is quantised to 30 fps for the VMD writer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .choreo import Choreo
from .mathx import (
    quat_from_euler_deg,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_slerp,
    smoothstep,
    swing_between,
)
from .music import MusicAnalysis
from .moves import get as get_move
from .rig import BASE_POSE, BONE_ORDER, JOINTS, clamp_pose

# leg geometry (MMD units) used by the groove <-> squat relation
L_THIGH = float(np.linalg.norm(np.array(JOINTS["knee.L"].rest) - np.array(JOINTS["leg.L"].rest)))
L_SHIN = float(np.linalg.norm(np.array(JOINTS["ankle.L"].rest) - np.array(JOINTS["knee.L"].rest)))
LEG_LEN = L_THIGH + L_SHIN

FPS = 30


# --------------------------------------------------------------------------- #
# pose state + forward kinematics
# --------------------------------------------------------------------------- #
@dataclass
class PoseState:
    local: dict[str, np.ndarray] = field(default_factory=dict)
    world: dict[str, np.ndarray] = field(default_factory=dict)
    pos: dict[str, np.ndarray] = field(default_factory=dict)
    root_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: int = 0
    beat: float = 0.0
    energy: float = 0.5


def fk(local: dict[str, np.ndarray], root_offset=(0, 0, 0), frame: int = 0, beat: float = 0.0, energy: float = 0.5) -> PoseState:
    st = PoseState(local={}, world={}, pos={}, root_offset=np.asarray(root_offset, float), frame=frame, beat=beat, energy=energy)
    for bone in BONE_ORDER:
        if JOINTS[bone].channel == "ik":
            continue
        j = JOINTS[bone]
        q = local.get(bone, np.array([1.0, 0.0, 0.0, 0.0]))
        st.local[bone] = q
        rest = np.asarray(j.rest, float)
        if j.parent is None:
            st.world[bone] = q
            st.pos[bone] = rest + st.root_offset
        else:
            parent = j.parent
            if parent not in st.world:  # defensive: unknown hierarchy
                st.world[bone] = q
                st.pos[bone] = rest + st.root_offset
                continue
            st.world[bone] = quat_mul(st.world[parent], q)
            rest_p = np.asarray(JOINTS[parent].rest, float)
            st.pos[bone] = st.pos[parent] + quat_rotate(st.world[parent], rest - rest_p)
    return st


def _rest_dir(bone: str) -> np.ndarray | None:
    """Rest-pose direction of a bone's primary axis (towards its first child)."""
    children = [n for n, j in JOINTS.items() if j.parent == bone]
    if not children:
        return None
    return np.asarray(JOINTS[children[0]].rest, float) - np.asarray(JOINTS[bone].rest, float)


# Cache rest directions
REST_DIR = {b: _rest_dir(b) for b in BONE_ORDER}


# --------------------------------------------------------------------------- #
# groove layer
# --------------------------------------------------------------------------- #
def groove(beat: float, energy: float, difficulty: float = 0.6) -> tuple[dict, np.ndarray]:
    """Beat-locked procedural layer.  Returns (euler deltas, root offset)."""
    e = (0.45 + 0.55 * float(np.clip(energy, 0.0, 1.0))) * (0.65 + 0.35 * float(np.clip(difficulty, 0, 1)))
    if e <= 1e-4:
        return {}, np.zeros(3)

    p = beat - math.floor(beat)
    down = 0.5 + 0.5 * math.cos(2 * math.pi * p)      # 1 on the beat, 0 between
    sway = math.sin(2 * math.pi * beat / 2.0)         # 2-beat weight shift
    bar = math.sin(2 * math.pi * beat / 4.0)          # 4-beat weight shift

    dip = 0.34 * e * down
    theta = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - dip / LEG_LEN))))
    knee_deg = 2.0 * theta
    leg_deg = -theta
    ankle_deg = -theta

    hip_roll = 6.0 * e * sway
    hip_yaw = 4.6 * e * bar
    d: dict[str, tuple[float, float, float]] = {}
    d["lower"] = (0.0, -hip_yaw * 0.45, hip_roll * 0.55)
    d["upper"] = (2.2 * e * down, -hip_yaw * 0.6, -hip_roll * 0.9)
    d["chest"] = (1.2 * e * down, -hip_yaw * 0.35, -hip_roll * 0.55)
    d["neck"] = (-1.4 * e * down, 0.0, hip_roll * 0.3)
    d["head"] = (-1.8 * e * down, hip_yaw * 0.3, hip_roll * 0.5)

    for side, s in (("L", 1), ("R", -1)):
        d[f"leg.{side}"] = (leg_deg, 0.0, -0.8 * e * sway * s)
        d[f"knee.{side}"] = (knee_deg, 0.0, 0.0)
        d[f"ankle.{side}"] = (ankle_deg, 0.0, 0.0)
        d[f"shoulder.{side}"] = (0.0, 0.0, 3.0 * e * down * s)
        # arms swing opposite to the hips, elbows breathe with the energy
        d[f"arm.{side}"] = (0.0, 11.0 * e * sway * s, -4.0 * e * down * s)
        d[f"elbow.{side}"] = (0.0, 10.0 * e * down * s, 0.0)

    root = np.array([0.16 * e * sway, -dip, 0.0])
    return d, root


# --------------------------------------------------------------------------- #
# score evaluation
# --------------------------------------------------------------------------- #
def slot_weight(beat: float, start: float, length: float, weight: float) -> float:
    if beat < start - 1e-6 or beat > start + length + 1e-6 or length <= 0:
        return 0.0
    p = (beat - start) / length
    attack = min(0.12, 0.5 / length)
    release = min(0.34, 0.8 / length)
    if p < attack:
        w = smoothstep(p / max(1e-6, attack))
    elif p > 1.0 - release:
        w = smoothstep((1.0 - p) / max(1e-6, release))
    else:
        w = 1.0
    return w * weight


def score_delta(choreo: Choreo, beat: float) -> tuple[dict, np.ndarray]:
    """Weighted sum of active move deltas, relative to BASE_POSE."""
    acc: dict[str, np.ndarray] = {}
    root = np.zeros(3)
    total = 0.0
    for s in choreo.slots:
        w = slot_weight(beat, s.beat, s.beats, s.weight)
        if w <= 1e-4:
            continue
        mv = get_move(s.move)
        phase = (beat - s.beat) / max(1e-6, s.beats)
        pose = mv.pose_at(phase)
        for j, v in pose.items():
            base = np.asarray(BASE_POSE[j], float)
            acc[j] = acc.get(j, np.zeros(3)) + w * (np.asarray(v, float) - base)
        root = root + w * np.asarray(mv.root_at(phase), float)
        total += w
    if total > 1.0:
        for j in acc:
            acc[j] /= total
        root /= total
    delta = {j: (float(v[0]), float(v[1]), float(v[2])) for j, v in acc.items()}
    return delta, root


def compose_pose(choreo: Choreo, analysis: MusicAnalysis, beat: float, energy: float) -> tuple[dict, np.ndarray]:
    delta, root_delta = score_delta(choreo, beat)
    g, g_root = groove(beat, energy, choreo.difficulty)
    pose: dict[str, tuple[float, float, float]] = {}
    for j, base in BASE_POSE.items():
        d = np.asarray(delta.get(j, (0.0, 0.0, 0.0)), float) + np.asarray(g.get(j, (0.0, 0.0, 0.0)), float)
        pose[j] = (float(base[0] + d[0]), float(base[1] + d[1]), float(base[2] + d[2]))
    return clamp_pose(pose), root_delta + g_root


# --------------------------------------------------------------------------- #
# two bone IK (leg) + foot locking
# --------------------------------------------------------------------------- #
def _solve_leg_ik(st: PoseState, side: str, target: np.ndarray, pole: np.ndarray) -> dict[str, np.ndarray]:
    """Return local quats for leg/knee/ankle that put the ankle on ``target``."""
    bone_leg, bone_knee, bone_ankle = f"leg.{side}", f"knee.{side}", f"ankle.{side}"
    hip = st.pos[bone_leg]
    knee = st.pos[bone_knee]
    ankle = st.pos[bone_ankle]
    l1 = float(np.linalg.norm(knee - hip))
    l2 = float(np.linalg.norm(ankle - knee))

    to_target = target - hip
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-6:
        return {}
    dist_c = min(max(dist, abs(l1 - l2) + 1e-3), (l1 + l2) * 0.9995)
    u = to_target / dist

    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    p = np.asarray(pole, float)
    p = p - np.dot(p, u) * u
    npn = np.linalg.norm(p)
    if npn < 1e-6:
        p = np.cross(u, np.array([0.0, 0.0, 1.0]))
        npn = np.linalg.norm(p)
        if npn < 1e-6:
            p = np.cross(u, np.array([1.0, 0.0, 0.0]))
            npn = np.linalg.norm(p)
    p = p / npn

    knee_new = hip + a * u + h * p
    dir_leg = knee_new - hip
    dir_shin = (hip + dist_c * u) - knee_new

    parent_world_leg = st.world[JOINTS[bone_leg].parent] if JOINTS[bone_leg].parent else np.array([1.0, 0, 0, 0])
    r_leg_world = swing_between(REST_DIR[bone_leg], dir_leg)
    q_leg = quat_mul(quat_inv(parent_world_leg), r_leg_world)

    world_leg = quat_mul(parent_world_leg, q_leg)
    r_shin_world = swing_between(REST_DIR[bone_knee], dir_shin)
    q_knee = quat_mul(quat_inv(world_leg), r_shin_world)

    world_knee = quat_mul(world_leg, q_knee)
    # keep the foot orientation the FK pass produced
    q_ankle = quat_mul(quat_inv(world_knee), st.world[bone_ankle])

    return {bone_leg: q_leg, bone_knee: q_knee, bone_ankle: q_ankle}


def _apply_subtree(st: PoseState, bone: str, overrides: dict[str, np.ndarray]) -> PoseState:
    """Re-run FK on the chain below ``bone`` after an IK override."""
    if not overrides:
        return st
    local = dict(st.local)
    local.update(overrides)
    idx = BONE_ORDER.index(bone)
    for b in BONE_ORDER[idx:]:
        if JOINTS[b].channel == "ik":
            continue
        j = JOINTS[b]
        q = local[b]
        st.local[b] = q
        if j.parent is None:
            st.world[b] = q
            st.pos[b] = np.asarray(j.rest, float) + st.root_offset
        else:
            st.world[b] = quat_mul(st.world[j.parent], q)
            st.pos[b] = st.pos[j.parent] + quat_rotate(st.world[j.parent], np.asarray(j.rest, float) - np.asarray(JOINTS[j.parent].rest, float))
    return st


def bake(analysis: MusicAnalysis, choreo: Choreo, fps: int = FPS, max_frames: int | None = None) -> list[PoseState]:
    n_frames = int(round(analysis.duration * fps))
    if max_frames:
        n_frames = min(n_frames, int(max_frames))
    states: list[PoseState] = []

    lock = {"L": None, "R": None}       # (x, z) world anchor
    lock_w = {"L": 0.0, "R": 0.0}       # smooth IK influence
    ema_energy = 0.35

    for f in range(n_frames + 1):
        t = f / fps
        beat = analysis.beat_at(t)
        sec = analysis.section_at_beat(max(0.0, beat))
        raw_e = 0.55 * sec.energy + 0.45 * analysis.energy_at_beat(max(0.0, beat))
        ema_energy += 0.22 * (raw_e - ema_energy)
        energy = float(np.clip(ema_energy, 0.05, 1.0))

        pose, root = compose_pose(choreo, analysis, beat, energy)
        local = {j: quat_from_euler_deg(*v) for j, v in pose.items()}
        st = fk(local, root, frame=f, beat=beat, energy=energy)

        # ---- keep the sole on the floor ---------------------------------
        # hip roll / sway can push one foot under the ground plane; raise the
        # whole body by the penetration depth instead of bending the leg.
        sole = min(
            st.pos["toe.L"][1] - 0.30,
            st.pos["toe.R"][1] - 0.30,
            st.pos["ankle.L"][1] - 0.86,
            st.pos["ankle.R"][1] - 0.86,
        )
        if sole < 0.0:
            lift = np.array([0.0, -float(sole), 0.0])
            for _k in st.pos:
                st.pos[_k] = st.pos[_k] + lift
            st.root_offset = st.root_offset + lift

        # ---- foot locking ------------------------------------------------
        for side, s in (("L", 1), ("R", -1)):
            ankle = st.pos[f"ankle.{side}"]
            toe = st.pos[f"toe.{side}"]
            in_contact = (ankle[1] < 1.25) and (toe[1] < 0.80)
            if in_contact:
                if lock[side] is None:
                    lock[side] = (float(ankle[0]), float(ankle[2]))
                else:
                    dx = ankle[0] - lock[side][0]
                    dz = ankle[2] - lock[side][1]
                    if math.hypot(dx, dz) > 2.2:  # a real step, re-anchor
                        lock[side] = (float(ankle[0]), float(ankle[2]))
                lock_w[side] = min(1.0, lock_w[side] + 0.20)
            else:
                lock_w[side] = max(0.0, lock_w[side] - 0.34)
                if lock_w[side] <= 0.0:
                    lock[side] = None

            if lock[side] is not None and lock_w[side] > 1e-3:
                target = np.array([lock[side][0], ankle[1], lock[side][1]])
                overrides = _solve_leg_ik(st, side, target, pole=np.array([0.35 * s, -0.2, 1.0]))
                if overrides:
                    blended = {}
                    for b, q in overrides.items():
                        blended[b] = quat_slerp(st.local[b], q, lock_w[side])
                    _apply_subtree(st, f"leg.{side}", blended)
        states.append(st)

    return states


# --------------------------------------------------------------------------- #
# VMD export
# --------------------------------------------------------------------------- #
def to_vmd(states: list[PoseState], model_name: str = "mmdance", fps: int = FPS, emit_ik: bool = True):
    from .mathx import to_mmd_quat, to_mmd_vec
    from .vmd import VmdMotion

    motion = VmdMotion(model_name=model_name)
    for st in states:
        root_world_q = st.world["root"]
        root_world_p = st.pos["root"]
        motion.add(
            JOINTS["root"].mmd,
            st.frame,
            position=tuple(to_mmd_vec(st.root_offset)),
            rotation=tuple(to_mmd_quat(st.local["root"])),
        )
        for bone in BONE_ORDER:
            if JOINTS[bone].channel == "ik" or bone == "root":
                continue
            q = to_mmd_quat(st.local[bone])
            motion.add(JOINTS[bone].mmd, st.frame, position=(0, 0, 0), rotation=tuple(q))

        # leg IK targets, expressed in the root's local frame
        if not emit_ik:
            continue
        inv_root = quat_inv(root_world_q)
        for side in ("L", "R"):
            ik = f"leg_ik.{side}"
            target = st.pos[f"ankle.{side}"]
            rel_rest = np.asarray(JOINTS[ik].rest, float) - np.asarray(JOINTS["root"].rest, float)
            local_target = quat_rotate(inv_root, target - root_world_p)
            offset = local_target - rel_rest
            motion.add(JOINTS[ik].mmd, st.frame, position=tuple(to_mmd_vec(offset)), rotation=(1.0, 0.0, 0.0, 0.0))
    return motion
