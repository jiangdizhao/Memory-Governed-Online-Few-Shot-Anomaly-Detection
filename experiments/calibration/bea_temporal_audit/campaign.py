"""Formal campaign completeness and expected accounting invariants."""
from __future__ import annotations
import math
from typing import Sequence
from .core import ARMS, AuditError, validate_rows
from .integrity import VerifiedTrajectory

CATEGORIES = ("audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper")
OLD = "40d61c3e948bdb89088f70845d2562444066d6e3"
NEW = "a3688e87ac165866fa4c4ddb3501f96258500534"
RELEASE = "52b045ea219eb270c427d30785da8e9b891c8eff2ba035b4430560bf17c962ae"
IDENTITY = "4d6841d048cb073d3b121ee025a3afd3619572b62afcf167081acbb6aa0f9912"
OLD_PLAN = "0239303a80395d8ad039f319abf5cdec09ee19e83481289ab88a42ee0af38fde"
COUNTS = dict(zip(CATEGORIES, (3815, 3785, 3790, 3805, 3835, 3770, 4470, 3800, 3775, 3790,
    3755, 3780, 3790, 3855, 3805, 3785, 3820, 3850, 3830, 3780, 3775, 3770, 3755,
    3740, 3820, 3860, 3765, 3915, 3750, 3750)))


def verify_formal_campaign(trajectories: Sequence[VerifiedTrajectory]) -> dict:
    expected = [(c, p) for c in CATEGORIES for p in ("P0", "P1", "P2")]
    by_pair = {(t.category, t.condition): t for t in trajectories}
    if len(trajectories) != 90 or len(by_pair) != 90 or set(by_pair) != set(expected):
        raise AuditError("the formal campaign requires exactly 90 registered trajectories")
    counts = {a: 0 for a in ARMS}
    suppressed = {a: 0 for a in ARMS}
    total = 0
    plans: dict[str, set[str]] = {OLD: set(), NEW: set()}
    runs: dict[str, set[str]] = {OLD: set(), NEW: set()}
    for i, pair in enumerate(expected):
        t = by_pair[pair]
        if (t.source_commit != (OLD if i < 36 else NEW) or
                t.expected_identity.get("source_commit") != t.source_commit or
                t.expected_identity.get("category") != pair[0] or t.expected_identity.get("condition") != pair[1] or
                t.expected_identity.get("dataset_release_sha256") != RELEASE or
                t.expected_identity.get("identity_manifest_sha256") != IDENTITY or
                len(t.rows) != COUNTS[pair[0]] or t.seal.expected_event_count != COUNTS[pair[0]]):
            raise AuditError("trajectory source commit differs from the cross-campaign prefix")
        plans[t.source_commit].add(str(t.expected_identity.get("execution_plan_sha256", "")))
        runs[t.source_commit].add(str(t.expected_identity.get("campaign_run_id", "")))
        total += len(t.rows)
        v = validate_rows(t.rows, cap=math.floor(len(t.rows) * .03))
        for arm in ARMS:
            counts[arm] += v["executed"][arm]
            suppressed[arm] += sum(int(row["query_actions"][arm]["cap_suppressed"]) for row in t.rows)
    if (plans[OLD] != {OLD_PLAN} or len(plans[NEW]) != 1 or OLD_PLAN in plans[NEW] or
            any(len(x) != 1 or "" in x for x in runs.values()) or
            total != 343755 or counts != {"F-B": 10272, "K-B": 10272, "F-M": 10121, "K-M": 10068} or suppressed["F-B"] != 3287 or suppressed["K-B"] != 3492):
        raise AuditError("formal campaign event/query totals differ from the registered invariants")
    return {"trajectory_count": 90, "category_count": 30, "event_count": total,
            "executed_queries": counts, "cap_suppressed": suppressed, "status": "PASS"}


class VerifiedCampaign:
    """All 90 validated once before any post-seal projection is permitted."""
    def __init__(self, trajectories: Sequence[VerifiedTrajectory]):
        self.trajectories = tuple(trajectories)
        self.integrity = verify_formal_campaign(self.trajectories)
        self.source_plan_sha256 = tuple(sorted({t.expected_identity["execution_plan_sha256"] for t in self.trajectories}))
