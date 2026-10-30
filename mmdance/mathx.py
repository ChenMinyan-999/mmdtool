"""Small quaternion / vector helpers (numpy only).

Convention
----------
* Quaternions are stored as ``[w, x, y, z]`` (scalar first).
* Angles in the public API are in **degrees**.
* Euler order is ``q = qz * qy * qx`` (X applied first, then Y, then Z).
* Canonical world space: Y up, character faces +Z, character's LEFT is +X,
  right-handed. ``to_mmd_*`` converts into MMD's left-handed space.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# basic quaternion ops
# --------------------------------------------------------------------------- #


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9 or abs(deg) < 1e-12:
        return quat_identity()
    axis = axis / n
    half = np.deg2rad(deg) * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return quat_identity()
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inv(q: np.ndarray) -> np.ndarray:
    n2 = float(np.dot(q, q))
    if n2 < 1e-12:
        return quat_identity()
    return quat_conj(q) / n2


def quat_from_euler_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """q = qz * qy * qx (degrees)."""
    qx = quat_from_axis_angle((1, 0, 0), rx)
    qy = quat_from_axis_angle((0, 1, 0), ry)
    qz = quat_from_axis_angle((0, 0, 1), rz)
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_rotate(q: np.ndarray, v) -> np.ndarray:
    return quat_to_matrix(q) @ np.asarray(v, dtype=float)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    return quat_normalize((np.sin((1 - t) * theta) / s) * a + (np.sin(t * theta) / s) * b)


def quat_angle_deg(q: np.ndarray) -> float:
    """Rotation magnitude of a quaternion, in degrees."""
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(q[0]), -1.0, 1.0))))


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle between two orientations, in degrees."""
    d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(np.rad2deg(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


def swing_between(v_from, v_to) -> np.ndarray:
    """Minimal-arc rotation taking direction ``v_from`` onto ``v_to``."""
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return quat_identity()
    a = a / na
    b = b / nb
    d = float(np.dot(a, b))
    if d > 0.999999:
        return quat_identity()
    if d < -0.999999:
        # 180 degrees: pick any perpendicular axis
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return quat_from_axis_angle(axis, 180.0)
    axis = np.cross(a, b)
    angle = np.rad2deg(np.arccos(np.clip(d, -1.0, 1.0)))
    return quat_from_axis_angle(axis, angle)


# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_sine(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return -(np.cos(np.pi * t) - 1.0) / 2.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# MMD conversion (right-handed canonical -> MMD left-handed)
# --------------------------------------------------------------------------- #

_FLIP_Z = True


def set_mmd_axis(flip_z: bool = True) -> None:
    """Flip this if the motion looks mirrored/backwards inside MMD."""
    global _FLIP_Z
    _FLIP_Z = bool(flip_z)


def to_mmd_vec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.array([v[0], v[1], -v[2]]) if _FLIP_Z else v.copy()


def to_mmd_quat(q: np.ndarray) -> np.ndarray:
    """RH quaternion -> LH quaternion (mirror about the z=0 plane)."""
    if not _FLIP_Z:
        return np.asarray(q, dtype=float).copy()
    return np.array([q[0], q[1], q[2], -q[3]])
