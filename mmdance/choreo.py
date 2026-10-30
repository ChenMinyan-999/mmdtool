"""L1 - the choreography score.

The score is the *contract* between the planner (an LLM, or the deterministic
rule planner below) and the synthesiser.  It is deliberately small, textual and
checkable::

    {
      "version": "1.0",
      "style": "idol-pop",
      "difficulty": 0.6,
      "bpm": 120.0,
      "beats": 48,
      "sections": [{"name": "chorus", "start_beat": 8, "end_beat": 32, "energy": 0.9}],
      "slots": [{"beat": 8, "move": "arms_wide", "beats": 4, "weight": 1.0, "note": ""}]
    }

Nothing in here is a joint angle, a frame number or a bone name - that is the
entire point.  It is what a language model can be trusted to produce, and what a
human can edit between two renders.
"""
from __future__ import annotations

import datetime as _dt
import json
import random
from dataclasses import dataclass, field, asdict

from . import moves as M
from .music import MusicAnalysis

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    beat: float
    move: str
    beats: float
    weight: float = 1.0
    note: str = ""

    @property
    def end(self) -> float:
        return self.beat + self.beats


@dataclass
class SectionScore:
    name: str
    start_beat: float
    end_beat: float
    energy: float


@dataclass
class Choreo:
    version: str = SCHEMA_VERSION
    style: str = "idol-pop"
    difficulty: float = 0.6
    bpm: float = 120.0
    beats: float = 0.0
    sections: list[SectionScore] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "style": self.style,
            "difficulty": round(self.difficulty, 3),
            "bpm": round(self.bpm, 3),
            "beats": round(self.beats, 3),
            "sections": [asdict(s) for s in self.sections],
            "slots": [asdict(s) for s in self.slots],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Choreo":
        return cls(
            version=str(d.get("version", SCHEMA_VERSION)),
            style=str(d.get("style", "idol-pop")),
            difficulty=float(d.get("difficulty", 0.6)),
            bpm=float(d.get("bpm", 120.0)),
            beats=float(d.get("beats", 0.0)),
            sections=[
                SectionScore(
                    str(s.get("name", "verse")),
                    float(s.get("start_beat", 0)),
                    float(s.get("end_beat", 0)),
                    float(s.get("energy", 0.6)),
                )
                for s in d.get("sections", [])
            ],
            slots=[
                Slot(
                    float(s.get("beat", 0)),
                    str(s.get("move", "")),
                    float(s.get("beats", 2)),
                    float(s.get("weight", 1.0)),
                    str(s.get("note", "")),
                )
                for s in d.get("slots", [])
            ],
            meta=dict(d.get("meta", {})),
        )

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path) -> "Choreo":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def sorted_slots(self) -> list[Slot]:
        return sorted(self.slots, key=lambda s: (s.beat, s.move))


# --------------------------------------------------------------------------- #
# validation - the only thing standing between an LLM and a broken VMD
# --------------------------------------------------------------------------- #
def validate(choreo: Choreo, strict: bool = True) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not (0.0 <= choreo.difficulty <= 1.0):
        errors.append(f"difficulty {choreo.difficulty} outside 0..1")
    if choreo.beats <= 0:
        errors.append("beats must be > 0")

    slots = choreo.sorted_slots()
    if not slots:
        errors.append("score has no slots")

    prev_end = -1e9
    for i, s in enumerate(slots):
        if s.move not in M.LIBRARY:
            errors.append(f"slot {i}: unknown move {s.move!r}")
            continue
        mv = M.LIBRARY[s.move]
        if s.beats <= 0:
            errors.append(f"slot {i} ({s.move}): duration must be > 0")
        if s.beat < -0.001:
            errors.append(f"slot {i} ({s.move}): negative start beat")
        if s.end > choreo.beats + 0.5:
            warnings.append(f"slot {i} ({s.move}): ends at beat {s.end:.1f} beyond score end")
        if not (0.0 <= s.weight <= 1.5):
            errors.append(f"slot {i} ({s.move}): weight {s.weight} outside 0..1.5")
        if abs(s.beats - mv.beats) > 0.26 and s.beats not in (2.0, 4.0):
            warnings.append(
                f"slot {i} ({s.move}): duration {s.beats} differs from library ({mv.beats})"
            )
        if s.beat + 0.01 < prev_end:
            overlap = prev_end - s.beat
            if overlap > 1.01:
                errors.append(
                    f"slot {i} ({s.move}): overlaps previous slot by {overlap:.2f} beats"
                )
            else:
                warnings.append(f"slot {i} ({s.move}): {overlap:.2f} beat crossfade")
        prev_end = max(prev_end, s.end)

        # energy fit
        lo, hi = mv.energy
        e = _energy_at(choreo, s.beat)
        if e < lo - 0.25 or e > hi + 0.25:
            warnings.append(
                f"slot {i} ({s.move}): section energy {e:.2f} outside move band {lo}-{hi}"
            )

    # coverage: a gap longer than a bar with no move is suspicious
    cursor = 0.0
    for s in slots:
        if s.beat - cursor > 4.0:
            warnings.append(f"gap of {s.beat - cursor:.1f} beats before beat {s.beat:.1f}")
        cursor = max(cursor, s.end)

    if strict and errors:
        return errors, warnings
    return errors, warnings


def _energy_at(choreo: Choreo, beat: float) -> float:
    for s in choreo.sections:
        if s.start_beat <= beat < s.end_beat:
            return s.energy
    return choreo.sections[-1].energy if choreo.sections else 0.6


# --------------------------------------------------------------------------- #
# rule based planner (deterministic, offline, always available)
# --------------------------------------------------------------------------- #
CALM = ["sway", "body_roll", "freeze"]
MID = ["sway", "arm_pump_r", "arm_pump_l", "point_side_r", "clap", "step_touch_r", "step_touch_l", "body_roll", "hip_bump_r"]
HOT = ["arms_wide", "point_up_r", "wave_arms", "kick_l", "kick_r", "clap", "pump_down", "turn_pose", "step_touch_r", "arm_pump_r", "arm_pump_l"]
BIG = ["arms_wide", "point_up_r", "wave_arms", "turn_pose", "kick_l", "kick_r"]


def plan(
    analysis: MusicAnalysis,
    style: str = "idol-pop",
    difficulty: float = 0.6,
    seed: int = 7,
) -> Choreo:
    """Turn a music analysis into a choreography score.

    Musical decisions made here (and mirrored in the LLM prompt):
      * phrases are 4 beats (one bar) or 2 beats (half bar)
      * every 8-beat phrase opens with a "big" move when the section is loud
      * left/right accents alternate so the weight never stays on one side
      * energy of the section gates the vocabulary
    """
    rng = random.Random(seed)
    n_beats = float(analysis.n_beats)
    sections = [
        SectionScore(s.name, s.start_beat, s.end_beat, s.energy) for s in analysis.sections
    ]

    slots: list[Slot] = []
    last_move = ""
    last_side = ""
    for sec in sections:
        e = max(0.05, min(1.0, sec.energy * (0.85 + 0.3 * difficulty)))
        start = int(round(sec.start_beat))
        end = int(round(sec.end_beat))
        start -= start % 4
        beat = start
        bar_index = start // 4
        while beat < end:
            bar_len = 4 if end - beat >= 4 else int(end - beat)
            if bar_len < 2:
                break
            phrase_start = (beat % 8) == 0
            if e < 0.45:
                pool, want = CALM, 4
            elif e < 0.75:
                pool, want = MID, (4 if phrase_start else 2)
            else:
                pool, want = (HOT if bar_len >= 4 else MID), (4 if phrase_start else 2)
            if bar_len >= 4 and (phrase_start or e >= 0.75) and e > 0.55:
                want = 4
            else:
                want = min(2, bar_len)

            k = 0
            while k < bar_len:
                n = min(want, bar_len - k)
                if n <= 0:
                    break
                name = _pick(rng, pool, last_move, last_side)
                mv = M.LIBRARY[name]
                # keep the slot inside the bar grid
                n = min(n, max(2, int(round(mv.beats))) if mv.beats >= 4 else 2)
                n = min(n, bar_len - k)
                if n < 2:
                    break
                side = ""
                if name.endswith("_r"):
                    side = "R"
                elif name.endswith("_l"):
                    side = "L"
                slots.append(
                    Slot(
                        beat=float(beat + k),
                        move=name,
                        beats=float(n),
                        weight=round(0.72 + 0.28 * min(1.0, e) * difficulty, 3),
                        note=f"{sec.name} e={e:.2f}",
                    )
                )
                last_move, last_side = name, side
                k += n
            beat += bar_len
            bar_index += 1

    # widen the last slot if the tail is ragged, and drop micro slots
    slots = [s for s in slots if s.beats >= 2]
    choreo = Choreo(
        style=style,
        difficulty=difficulty,
        bpm=analysis.bpm,
        beats=n_beats,
        sections=sections,
        slots=slots,
        meta={
            "planner": "rule",
            "seed": seed,
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "source": analysis.source,
        },
    )
    return choreo


def _pick(rng: random.Random, pool: list[str], last_move: str, last_side: str) -> str:
    for _ in range(24):
        name = rng.choice(pool)
        if name == last_move:
            continue
        side = "R" if name.endswith("_r") else ("L" if name.endswith("_l") else "")
        if side and side == last_side:
            continue
        return name
    return rng.choice(pool)


# --------------------------------------------------------------------------- #
# LLM planner (optional, OpenAI compatible endpoint)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a professional K-pop/idol choreographer writing for a MikuMikuDance avatar.
You do NOT produce joint angles. You produce a choreography score: a list of
"slots" that place moves from the given library onto the beat grid.

Hard rules:
1. Output ONLY a JSON object, no prose, no markdown fences.
2. Every slot must use a "move" name from the library. Never invent names.
3. Slots must be sequential and must not overlap by more than 1 beat.
4. Slot lengths are 2 or 4 beats and must stay inside the section that
   motivates them. Prefer 4-beat phrases; use 2-beat accents to fill gaps.
5. Match section energy to the move's energy band.
6. Alternate left/right accents; never repeat the same move back to back.
7. On the first beat of an 8-beat phrase, put a "big" move (tag: big) when the
   section energy is above 0.55.
8. Keep the total slot duration within 2% of the track length.

JSON shape:
{"version":"1.0","style":"<style>","difficulty":<0..1>,"bpm":<float>,"beats":<float>,
 "sections":[{"name":str,"start_beat":float,"end_beat":float,"energy":float}],
 "slots":[{"beat":float,"move":str,"beats":float,"weight":float,"note":str}]}"""


def build_llm_prompt(analysis: MusicAnalysis, style: str, difficulty: float) -> tuple[str, str]:
    catalog = M.catalog()
    lib_lines = [
        f"- {m['name']}: {m['beats']} beats, channels={','.join(m['channels'])}, "
        f"energy={m['energy'][0]}-{m['energy'][1]}, tags={','.join(m['tags'])}. "
        f"{m['description']}"
        for m in catalog
    ]
    music = analysis.to_dict()
    music.pop("beat_energy", None)  # keep the prompt small
    user = f"""Style: {style}
Difficulty: {difficulty}

MUSIC ANALYSIS
{json.dumps(music, ensure_ascii=False)}

MOVE LIBRARY
{chr(10).join(lib_lines)}

Write the choreography score for this track."""
    return SYSTEM_PROMPT, user


def plan_with_llm(
    analysis: MusicAnalysis,
    style: str = "idol-pop",
    difficulty: float = 0.6,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    retries: int = 2,
) -> Choreo:
    """Ask an LLM for a score, validate it, and repair once on failure.

    This is the *only* place the pipeline needs a network / API key.  The
    deterministic :func:`plan` is used as the fallback so the demo always runs.
    """
    import os
    import urllib.request

    api_key = api_key or os.environ.get("MMDANCE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "no API key: set MMDANCE_API_KEY (or OPENAI_API_KEY), or use the rule planner"
        )
    base_url = (base_url or os.environ.get("MMDANCE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    system, user = build_llm_prompt(analysis, style, difficulty)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for attempt in range(retries + 1):
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        try:
            choreo = Choreo.from_dict(json.loads(_strip_fences(text)))
        except Exception as exc:  # noqa: BLE001
            last_error = f"unparsable JSON: {exc}"
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"That is not valid JSON ({exc}). Return only the JSON object."})
            continue
        errors, _ = validate(choreo)
        if not errors:
            choreo.meta.update({"planner": "llm", "model": model, "attempt": attempt})
            return choreo
        last_error = "; ".join(errors[:5])
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": "Your score has errors, fix them and return the full JSON again:\n- "
                + "\n- ".join(errors[:8]),
            }
        )
    raise RuntimeError(f"LLM planner failed after {retries + 1} attempts: {last_error}")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()
