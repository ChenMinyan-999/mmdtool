"""mmdance - LLM-as-choreographer pipeline that bakes MMD `.vmd` motions.

Pipeline
--------
L0 music      : deterministic analysis  -> beats / sections / energy
L1 choreo     : LLM (or rule planner)   -> choreography score (JSON)
L2 synthesis  : motion library + groove + IK / foot lock
L3 bake       : retarget + 30 fps quantise -> VMD
L4 verify     : foot slide, penetration, balance, limits, beat alignment
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
