"""Minimal but spec-correct VMD (Vocaloid Motion Data 0002) reader/writer.

Layout (little endian)
----------------------
header          30 bytes  "Vocaloid Motion Data 0002" + NUL padding
model name      20 bytes  cp932, NUL padded
bone frames     uint32 n, then n * 111 bytes
morph frames    uint32 n, then n * 23 bytes
camera frames   uint32 n, then n * 61 bytes
light frames    uint32 n, then n * 28 bytes
self shadow     uint32 n, then n * 5 bytes
model display   uint32 n, then n * 1 byte     (optional)
IK on/off       uint32 n, then n * 4 bytes    (optional)

bone frame = name[15] uint32 frame, float3 position, float4 rotation(w,x,y,z),
             64 bytes bezier interpolation.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


HEADER = b"Vocaloid Motion Data 0002"
ENCODING = "cp932"

# 20/127 control points on all four channels = linear interpolation.
# Every 4-byte group ends up holding (20, 20, 20, 20), which is linear under any
# of the two common readings of the 64-byte block.
LINEAR_INTERP = bytes([20] * 64)


def _pad(text: str, size: int, encoding: str = ENCODING) -> bytes:
    raw = text.encode(encoding, errors="replace")[: size - 1]
    return raw + b"\x00" * (size - len(raw))


@dataclass
class BoneKey:
    bone: str
    frame: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z


@dataclass
class MorphKey:
    morph: str
    frame: int
    weight: float


@dataclass
class VmdMotion:
    model_name: str = "mmdance"
    bones: list[BoneKey] = field(default_factory=list)
    morphs: list[MorphKey] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def add(self, bone: str, frame: int, position=(0, 0, 0), rotation=(1, 0, 0, 0)) -> None:
        self.bones.append(
            BoneKey(
                bone=bone,
                frame=int(frame),
                position=(float(position[0]), float(position[1]), float(position[2])),
                rotation=(
                    float(rotation[0]),
                    float(rotation[1]),
                    float(rotation[2]),
                    float(rotation[3]),
                ),
            )
        )

    def add_morph(self, morph: str, frame: int, weight: float) -> None:
        self.morphs.append(MorphKey(morph, int(frame), float(weight)))

    # ------------------------------------------------------------------ #
    def to_bytes(self) -> bytes:
        out = bytearray()
        out += HEADER + b"\x00" * (30 - len(HEADER))
        out += _pad(self.model_name, 20)

        bones = sorted(self.bones, key=lambda k: (k.bone, k.frame))
        out += struct.pack("<I", len(bones))
        for k in bones:
            out += _pad(k.bone, 15)
            out += struct.pack("<I", k.frame)
            out += struct.pack("<3f", *k.position)
            out += struct.pack("<4f", *k.rotation)
            out += LINEAR_INTERP

        morphs = sorted(self.morphs, key=lambda k: (k.morph, k.frame))
        out += struct.pack("<I", len(morphs))
        for m in morphs:
            out += _pad(m.morph, 15)
            out += struct.pack("<If", m.frame, m.weight)

        out += struct.pack("<I", 0)  # camera
        out += struct.pack("<I", 0)  # light
        out += struct.pack("<I", 0)  # self shadow
        out += struct.pack("<I", 0)  # model display
        out += struct.pack("<I", 0)  # IK on/off
        return bytes(out)

    def save(self, path) -> int:
        data = self.to_bytes()
        with open(path, "wb") as fh:
            fh.write(data)
        return len(data)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path) -> "VmdMotion":
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())

    @classmethod
    def from_bytes(cls, data: bytes) -> "VmdMotion":
        if data[: len(HEADER)] != HEADER:
            raise ValueError("not a VMD 0002 file")
        off = 30
        model_name = data[off : off + 20].split(b"\x00")[0].decode(ENCODING, "replace")
        off += 20
        motion = cls(model_name=model_name)

        (n_bone,) = struct.unpack_from("<I", data, off)
        off += 4
        for _ in range(n_bone):
            name = data[off : off + 15].split(b"\x00")[0].decode(ENCODING, "replace")
            (frame,) = struct.unpack_from("<I", data, off + 15)
            pos = struct.unpack_from("<3f", data, off + 19)
            rot = struct.unpack_from("<4f", data, off + 31)
            off += 111
            motion.bones.append(BoneKey(name, int(frame), pos, rot))

        (n_morph,) = struct.unpack_from("<I", data, off)
        off += 4
        for _ in range(n_morph):
            name = data[off : off + 15].split(b"\x00")[0].decode(ENCODING, "replace")
            frame, weight = struct.unpack_from("<If", data, off + 15)
            off += 23
            motion.morphs.append(MorphKey(name, int(frame), float(weight)))
        return motion

    # ------------------------------------------------------------------ #
    def bone_names(self) -> list[str]:
        seen = []
        for k in self.bones:
            if k.bone not in seen:
                seen.append(k.bone)
        return seen

    def summary(self) -> dict:
        frames = [k.frame for k in self.bones]
        return {
            "model_name": self.model_name,
            "bones": len(self.bone_names()),
            "bone_keys": len(self.bones),
            "morph_keys": len(self.morphs),
            "frames": (min(frames), max(frames)) if frames else (0, 0),
            "duration_s": (max(frames) / 30.0) if frames else 0.0,
        }
