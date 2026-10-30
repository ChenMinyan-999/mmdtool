"""Unit / integration tests (stdlib unittest, no extra dependencies).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmdance import choreo as C  # noqa: E402
from mmdance import mathx as MX  # noqa: E402
from mmdance import moves as MV  # noqa: E402
from mmdance import music as MU  # noqa: E402
from mmdance import rig  # noqa: E402
from mmdance import synth as SY  # noqa: E402
from mmdance import validate as VD  # noqa: E402
from mmdance.vmd import VmdMotion  # noqa: E402

DEMO = ROOT / "examples" / "demo_song.wav"


class TestQuatMath(unittest.TestCase):
    def test_identity(self):
        self.assertTrue(np.allclose(MX.quat_from_euler_deg(0, 0, 0), [1, 0, 0, 0]))

    def test_rotation_axis(self):
        q = MX.quat_from_euler_deg(90, 0, 0)
        v = MX.quat_rotate(q, (0, 1, 0))
        self.assertTrue(np.allclose(v, [0, 0, 1], atol=1e-6))

    def test_swing_between(self):
        q = MX.swing_between((1, 0, 0), (0, 1, 0))
        self.assertTrue(np.allclose(MX.quat_rotate(q, (1, 0, 0)), [0, 1, 0], atol=1e-6))

    def test_swing_opposite(self):
        q = MX.swing_between((0, 1, 0), (0, -1, 0))
        self.assertTrue(np.allclose(MX.quat_rotate(q, (0, 1, 0)), [0, -1, 0], atol=1e-6))

    def test_compose_is_normalised(self):
        q = MX.quat_from_euler_deg(33, -21, 77)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)


class TestRig(unittest.TestCase):
    def test_bone_names_are_mmd(self):
        names = [j.mmd for j in rig.JOINT_LIST]
        self.assertIn("センター", names)
        self.assertIn("左足ＩＫ", names)  # full width IK
        self.assertEqual(len(names), len(set(names)), "duplicate MMD bone names")

    def test_mmd_name_fits_15_bytes(self):
        for j in rig.JOINT_LIST:
            self.assertLessEqual(len(j.mmd.encode("cp932")), 14, j.mmd)

    def test_base_pose_within_limits(self):
        for name, v in rig.BASE_POSE.items():
            for axis, val in enumerate(v):
                lo, hi = rig.LIMITS.get((name, axis), rig.DEFAULT_LIMIT)
                self.assertGreaterEqual(val, lo - 1e-6, f"{name}[{axis}]")
                self.assertLessEqual(val, hi + 1e-6, f"{name}[{axis}]")


class TestVmd(unittest.TestCase):
    def test_roundtrip(self):
        m = VmdMotion(model_name="テスト")
        m.add("センター", 0, position=(1.0, 2.0, -3.0), rotation=(1.0, 0.0, 0.0, 0.0))
        m.add("左足ＩＫ", 30, position=(0.1, -0.2, 0.3), rotation=(0.7071, 0.0, 0.7071, 0.0))
        m.add_morph("まばたき", 10, 0.5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.vmd"
            size = m.save(p)
            raw = p.read_bytes()
            self.assertEqual(raw[:24], b"Vocaloid Motion Data 000")
            self.assertEqual(raw[30:50].split(b"\x00")[0].decode("cp932"), "テスト")
            self.assertEqual(len(raw), size)
            back = VmdMotion.load(p)
        self.assertEqual(back.model_name, "テスト")
        self.assertEqual(len(back.bones), 2)
        k = [b for b in back.bones if b.bone == "左足ＩＫ"][0]
        self.assertEqual(k.frame, 30)
        self.assertTrue(np.allclose(k.position, (0.1, -0.2, 0.3), atol=1e-6))
        self.assertTrue(np.allclose(k.rotation, (0.7071, 0.0, 0.7071, 0.0), atol=1e-5))
        self.assertEqual(back.morphs[0].weight, 0.5)

    def test_record_size(self):
        m = VmdMotion()
        m.add("センター", 0)
        raw = m.to_bytes()
        # header(30) + name(20) + count(4) + 1 key(111) + 4 zero counts + 2 empty sections
        self.assertEqual(len(raw) - (30 + 20 + 4), 111 + 4 * 4 + 8)


def _analysis():
    if not DEMO.exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_music.py"), str(DEMO)], check=True)
    return MU.analyze(DEMO)


class TestMusic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = _analysis()

    def test_tempo(self):
        self.assertAlmostEqual(self.a.bpm, 120.0, delta=1.5)

    def test_beat_grid_covers_track(self):
        self.assertGreaterEqual(self.a.n_beats, 44)
        self.assertLess(abs(float(self.a.beat_times[0])), 0.25, "grid should start at t~0")
        self.assertLess(float(self.a.beat_times[-1]), self.a.duration + 0.1)

    def test_sections(self):
        self.assertGreaterEqual(len(self.a.sections), 2)
        self.assertAlmostEqual(self.a.sections[0].start_beat, 0.0)
        self.assertAlmostEqual(self.a.sections[-1].end_beat, float(self.a.n_beats), places=6)
        for s in self.a.sections:
            self.assertLess(s.start_beat, s.end_beat)

    def test_beat_times_monotonic(self):
        d = np.diff(self.a.beat_times)
        self.assertTrue(np.all(d > 0))


class TestChoreo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = _analysis()
        cls.score = C.plan(cls.a, seed=7)

    def test_plan_valid(self):
        errors, _ = C.validate(self.score)
        self.assertEqual(errors, [])

    def test_plan_is_sequential(self):
        slots = self.score.sorted_slots()
        for prev, cur in zip(slots, slots[1:]):
            self.assertGreaterEqual(cur.beat, prev.end - 0.01)

    def test_plan_uses_real_moves(self):
        for s in self.score.slots:
            self.assertIn(s.move, MV.LIBRARY)

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            self.score.save(p)
            back = C.Choreo.load(p)
            self.assertEqual(len(back.slots), len(self.score.slots))
            self.assertEqual(back.slots[0].move, self.score.slots[0].move)

    def test_unknown_move_rejected(self):
        bad = C.Choreo(beats=8, slots=[C.Slot(0, "moonwalk", 2)])
        errors, _ = C.validate(bad)
        self.assertTrue(any("unknown move" in e for e in errors))

    def test_overlap_rejected(self):
        bad = C.Choreo(beats=8, slots=[C.Slot(0, "sway", 4), C.Slot(1, "clap", 2)])
        errors, _ = C.validate(bad)
        self.assertTrue(any("overlap" in e for e in errors))

    def test_llm_prompt_mentions_library(self):
        system, user = C.build_llm_prompt(self.a, "idol-pop", 0.6)
        self.assertIn("JSON", system)
        for name in ("sway", "arms_wide", "kick_l"):
            self.assertIn(name, user)


class TestMoves(unittest.TestCase):
    def test_catalog_poses_finite(self):
        for name, mv in MV.LIBRARY.items():
            for p in np.linspace(0, 1, 7):
                pose = mv.pose_at(float(p))
                self.assertEqual(set(pose), set(rig.BASE_POSE), name)
                for v in pose.values():
                    self.assertTrue(all(np.isfinite(x) for x in v), name)
                    self.assertLess(max(abs(x) for x in v), 400, name)

    def test_accent_moves_start_neutral(self):
        # accent moves must start from the base pose so blends have no jump;
        # "hold" style moves (clap, freeze, wave_arms) deliberately do not.
        for name in ("kick_l", "kick_r", "point_up_r", "arms_wide", "pump_down"):
            pose = MV.get(name).pose_at(0.0)
            diff = max(abs(pose[j][i] - rig.BASE_POSE[j][i]) for j in pose for i in range(3))
            self.assertLess(diff, 1e-6, f"{name} does not start neutral")


class TestSynthAndChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = _analysis()
        cls.score = C.plan(cls.a, seed=7)
        cls.states = SY.bake(cls.a, cls.score, max_frames=150)

    def test_frame_count(self):
        self.assertEqual(len(self.states), 151)

    def test_unique_bone_keys(self):
        motion = SY.to_vmd(self.states)
        seen = set()
        for k in motion.bones:
            key = (k.bone, k.frame)
            self.assertNotIn(key, seen, f"duplicate keyframe {key}")
            seen.add(key)

    def test_quaternions_unit(self):
        for st in self.states[::17]:
            for b, q in st.local.items():
                self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=5, msg=b)

    def test_no_penetration(self):
        for st in self.states:
            self.assertGreater(st.pos["toe.L"][1], 0.25, f"frame {st.frame}")
            self.assertGreater(st.pos["toe.R"][1], 0.25, f"frame {st.frame}")

    def test_foot_lock_works(self):
        # while a foot is planted, its horizontal position must not drift
        drift = 0.0
        prev = {}
        for st in self.states:
            for side in ("L", "R"):
                a = st.pos[f"ankle.{side}"]
                if a[1] < 1.05 and side in prev and prev[side][2]:
                    drift = max(drift, float(np.hypot(a[0] - prev[side][0], a[2] - prev[side][1])))
                prev[side] = (float(a[0]), float(a[2]), bool(a[1] < 1.05))
        self.assertLess(drift, 0.25, "planted foot drifted more than 2 cm in one frame")

    def test_report_shape(self):
        rep = VD.check(self.states, choreo=self.score, analysis=self.a)
        names = {m["name"] for m in rep["metrics"]}
        self.assertEqual(
            names,
            {"foot_slide", "ground_penetration", "balance_outside", "joint_limit_hits", "beat_alignment", "snap"},
        )
        self.assertIn(rep["overall"], {"PASS", "WARN", "FAIL"})


class TestEndToEnd(unittest.TestCase):
    def test_cli_build_no_render(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            code = __import__("mmdance.cli", fromlist=["main"]).main(
                ["demo", "--out", str(out), "--no-render", "--seconds", "4"]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "motion.vmd").exists())
            self.assertTrue((out / "score.json").exists())
            rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIn("metrics", rep)
            size = (out / "motion.vmd").stat().st_size
            self.assertGreater(size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
