"""R2-L2E label-blind H-SOURCE partition proposal helpers.

This module consumes caller-supplied SHA-256 source-group identities and
five-view counts only. It never discovers or opens dataset files and never
accepts labels, paths, images, features, masks, or score values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence

from experiments.calibration.r2_l2a_allclass_inventory import (
    CANONICAL_CATEGORY_IDS,
    GROUPING_UNIT,
    IDENTITY_CONTRACT_VERSION,
    PINNED_SOURCE_SHA256,
)
from experiments.calibration.r2_l2b_role_certification import (
    PROTECTED_DEFAULT,
    ROLE_NAMES,
    SOURCE_GROUP_DIGEST_ALGORITHM,
    ResearcherApproval,
    SourceGroupMember,
)
from experiments.calibration.r2_l2d_data_role_adjudication import (
    PARTITION_RANKING_ALGORITHM,
)


SCHEMA = "r2_l2e.h_source_partition_proposal.v1"
POLICY_SCHEMA = "r2_l2e.h_source_partition_policy.v1"
PARTITION_SEED = 20261101
EXPOSED_DEVELOPMENT_CATEGORIES = (
    "audiojack", "regulator", "rolled_strip_base", "switch",
)
PROTECTED_CATEGORIES = tuple(PROTECTED_DEFAULT)
VIEWS_PER_SOURCE_GROUP = 5
A_MIN_PROTOCOL_MATCHED_GROUPS = 68
A_MIN_FULL_GEOMETRY_GROUPS = 100
BC_MIN_GROUPS_PER_ROLE = 200
BC_MIN_GROUPS_PER_OUTCOME_CLASS = 100
G2_MIN_QUERIED_GROUPS_PER_OUTCOME_CLASS = 20
Q_A_TS_QUERY_FRACTION = "0.03"
Q_A_TS_TIME_BINS = 16
Q_A_TS_SEED = 37
TEMPORAL_ORDER_SEED = 20261102
BURSTY_BLOCK_GROUP_COUNT = 50
TEMPORAL_ORDER_RANKING_ALGORITHM = "sha256:r2-l2e-temporal-order.v1"


POLICY_PAYLOAD = {
    "schema": POLICY_SCHEMA,
    "version": "1.0.0-proposed",
    "strategy": "H-SOURCE",
    "catalog_source_sha256": PINNED_SOURCE_SHA256,
    "identity_contract_version": IDENTITY_CONTRACT_VERSION,
    "grouping_unit": GROUPING_UNIT,
    "group_digest_algorithm": SOURCE_GROUP_DIGEST_ALGORITHM,
    "views_per_source_group": VIEWS_PER_SOURCE_GROUP,
    "ranking_algorithm": PARTITION_RANKING_ALGORITHM,
    "partition_seed": PARTITION_SEED,
    "categories": list(CANONICAL_CATEGORY_IDS),
    "role_rules": {
        "dispositions": {
            "A_required_all_categories": list(CANONICAL_CATEGORY_IDS),
            "B_not_applicable_categories": list(PROTECTED_CATEGORIES),
            "C_not_applicable_categories": list(EXPOSED_DEVELOPMENT_CATEGORIES),
        },
        "A": {
            "population": "all_authorized_train_normal_source_groups",
            "disposition": "REQUIRED_ALL_30",
            "minimum_groups_for_four_supports_plus_64_C64": A_MIN_PROTOCOL_MATCHED_GROUPS,
            "minimum_groups_for_full_geometry": A_MIN_FULL_GEOMETRY_GROUPS,
        },
        "B": {
            "population": "authorized_test_source_groups_for_development",
            "ordinary_category_quota": "floor(3*n/5) fresh groups; prior-exposed groups also remain in B",
            "exposed_development_categories": list(EXPOSED_DEVELOPMENT_CATEGORIES),
            "protected_not_applicable_categories": list(PROTECTED_CATEGORIES),
            "minimum_groups_per_role": BC_MIN_GROUPS_PER_ROLE,
            "minimum_groups_per_outcome_class": BC_MIN_GROUPS_PER_OUTCOME_CLASS,
        },
        "C": {
            "population": "sealed_authorized_test_source_groups",
            "ordinary_category_quota": "n - floor(3*n/5) fresh groups",
            "not_applicable_exposed_development_categories": list(EXPOSED_DEVELOPMENT_CATEGORIES),
            "protected_confirmation_only_categories": list(PROTECTED_CATEGORIES),
            "minimum_groups_per_role": BC_MIN_GROUPS_PER_ROLE,
            "minimum_groups_per_outcome_class": BC_MIN_GROUPS_PER_OUTCOME_CLASS,
            "outcomes": "independent sealed evaluator only after method and analysis freeze",
        },
    },
    "source_group_rules": {
        "all_camera_views_stay_together": True,
        "train_test_identity_overlap": "A has priority; exclude overlapping Test group from B/C and record count",
        "prior_exposure": "exclude from C; assign to B except protected categories, which block for policy review",
        "cross_category_or_role_overlap": "reject",
        "insufficient_groups": "fail closed; do not change ratio, seed, or merge categories",
    },
    "support_c64": {
        "selection_population": "A only",
        "support_images_per_category_and_seed": 4,
        "additional_c64_images_per_category_and_seed": 64,
        "one_image_per_source_group": True,
        "support_c64_source_groups_disjoint": True,
        "role_membership_shared_across_support_seeds": True,
        "c64_is_strict_four_shot_total": False,
    },
    "temporal_stream": {
        "query_protocol": "Q-A-TS-v1",
        "query_fraction_of_image_events": Q_A_TS_QUERY_FRACTION,
        "time_bins": Q_A_TS_TIME_BINS,
        "query_seed": Q_A_TS_SEED,
        "order_stress_seed": TEMPORAL_ORDER_SEED,
        "order_ranking_algorithm": TEMPORAL_ORDER_RANKING_ALGORITHM,
        "bursty_block_groups": BURSTY_BLOCK_GROUP_COUNT,
        "maximum_queries_per_official_sample_key": 1,
        "minimum_queried_groups_per_outcome_class_for_G2": G2_MIN_QUERIED_GROUPS_PER_OUTCOME_CLASS,
        "label_use_for_order": "offline evaluator only; controller receives no future labels",
        "stress_orders": ["RANDOM", "NORMAL_FIRST", "DEFECT_FIRST", "BURSTY_50_GROUP"],
        "same_event_multiset": True,
    },
    "g1_g2_boundary": {
        "g1a": "49 fixed cell centers on R1 28x28 patch grid; source-disjoint normal geometry",
        "g1b": "native C0 Top-32 under category-specific Fixed Support and category-specific M0 state",
        "g2": "causal queried Correction Anchor to strictly future native patch study",
        "g1_substitutes_for_g2": False,
    },
    "claims": {
        "supported": "within-category source-group holdout under H-SOURCE",
        "unseen_category_generalization": False,
        "h_class_policy": "separately versioned and separately authorized if required",
    },
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


POLICY_SHA256 = canonical_sha256(POLICY_PAYLOAD)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PartitionProtocolError(ValueError):
    """A proposed role partition or its authorization boundary is invalid."""


@dataclass(frozen=True)
class CategoryPartitionProposal:
    schema: str
    category_id: str
    policy_sha256: str
    source_manifest_sha256: str
    seed: int
    ranking_algorithm: str
    status: str
    a_status: str
    b_status: str
    c_status: str
    a_full_geometry_status: str
    a_group_sha256s: tuple[str, ...]
    b_group_sha256s: tuple[str, ...]
    c_group_sha256s: tuple[str, ...]
    candidate_a_group_count: int
    candidate_test_group_count: int | None
    excluded_a_test_overlap_count: int
    prior_exposed_group_count: int
    fresh_test_group_count: int | None
    proposed_b_group_count: int
    proposed_c_group_count: int
    b_quota_on_fresh_groups: int | None
    c_quota_on_fresh_groups: int | None
    contains_raw_sample_keys: bool
    contains_image_paths: bool
    contains_labels_or_masks: bool
    proposal_sha256: str


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PartitionProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _member_hashes(
    members: Sequence[SourceGroupMember], *, name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for member in members:
        if not isinstance(member, SourceGroupMember):
            raise PartitionProtocolError(f"{name} contains an incompatible source-group member")
        digest = _require_sha256(member.group_sha256, f"{name} source-group digest")
        if (isinstance(member.image_view_count, bool)
                or not isinstance(member.image_view_count, int)
                or member.image_view_count != VIEWS_PER_SOURCE_GROUP):
            raise PartitionProtocolError(
                f"{name} groups must retain exactly {VIEWS_PER_SOURCE_GROUP} camera views")
        normalized.append(digest)
    if len(normalized) != len(set(normalized)):
        raise PartitionProtocolError(f"{name} contains duplicate source groups")
    return tuple(sorted(normalized))


def _rank_key(category_id: str, group_sha256: str) -> tuple[str, str]:
    payload = (
        f"{PARTITION_RANKING_ALGORITHM}\0{POLICY_SHA256}\0"
        f"{category_id}\0{PARTITION_SEED}\0{group_sha256}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest(), group_sha256


def _proposal_body(proposal: CategoryPartitionProposal) -> dict[str, object]:
    body = asdict(proposal)
    body.pop("proposal_sha256")
    return body


def _finish_proposal(**fields: object) -> CategoryPartitionProposal:
    partial = CategoryPartitionProposal(
        schema=SCHEMA,
        policy_sha256=POLICY_SHA256,
        seed=PARTITION_SEED,
        ranking_algorithm=PARTITION_RANKING_ALGORITHM,
        contains_raw_sample_keys=False,
        contains_image_paths=False,
        contains_labels_or_masks=False,
        proposal_sha256="",
        **fields,
    )
    return CategoryPartitionProposal(
        **{**asdict(partial), "proposal_sha256": canonical_sha256(_proposal_body(partial))}
    )


def propose_category_partition(
    category_id: str,
    train_normal_a_members: Sequence[SourceGroupMember] | None,
    test_members: Sequence[SourceGroupMember] | None,
    *,
    source_manifest_sha256: str,
    test_identity_scope_verified: bool = False,
    protected_test_scope_verified: bool = False,
    exposure_registry_verified: bool = False,
    prior_exposed_group_sha256s: Sequence[str] = (),
    policy_sha256: str = POLICY_SHA256,
    seed: int = PARTITION_SEED,
) -> CategoryPartitionProposal:
    """Create a hashed-ID-only H-SOURCE proposal; never certify membership.

    Authorization booleans are caller assertions made only after the existing
    L2B access-scope and signature validators succeed. This pure function does
    not authenticate authorizations or read data.
    """
    if category_id not in CANONICAL_CATEGORY_IDS:
        raise PartitionProtocolError("category is outside the pinned public catalog")
    _require_sha256(source_manifest_sha256, "combined source-manifest SHA-256")
    if policy_sha256 != POLICY_SHA256 or seed != PARTITION_SEED:
        raise PartitionProtocolError("policy hash or seed differs from the frozen L2E proposal")

    a_candidates = () if train_normal_a_members is None else _member_hashes(
        train_normal_a_members, name="A population")
    a_count = 0 if train_normal_a_members is None else len(train_normal_a_members)
    if train_normal_a_members is None:
        a_status = full_status = "PENDING_AUTHORIZED_A_IDENTITY_RECEIPT"
        a_groups: tuple[str, ...] = ()
    elif not a_candidates:
        a_status = full_status = "PENDING_MISSING_A_RECEIPT_NOT_EMPTY_CERTIFIED"
        a_groups = ()
    elif len(a_candidates) < A_MIN_PROTOCOL_MATCHED_GROUPS:
        a_status = "PENDING_INSUFFICIENT_GROUPS_FOR_4_PLUS_64"
        full_status = "PENDING_INSUFFICIENT_GROUPS"
        a_groups = ()
    else:
        a_groups = a_candidates
        a_status = "PROPOSED_REQUIRES_INDEPENDENT_ROLE_REVIEW"
        full_status = (
            "PROPOSED_REQUIRES_INDEPENDENT_ROLE_REVIEW"
            if len(a_groups) >= A_MIN_FULL_GEOMETRY_GROUPS
            else "PENDING_INSUFFICIENT_GROUPS_FOR_FULL_GEOMETRY"
        )

    def result(
        status: str, *, b_status: str, c_status: str,
        test_count: int | None = None, a_overlap: int = 0,
        prior_count: int = 0, fresh_count: int | None = None,
        b_count: int = 0, c_count: int = 0, b_quota: int | None = None,
        c_quota: int | None = None, b_groups: tuple[str, ...] = (),
        c_groups: tuple[str, ...] = (),
    ) -> CategoryPartitionProposal:
        proposal = _finish_proposal(
            category_id=category_id,
            source_manifest_sha256=source_manifest_sha256,
            status=status,
            a_status=a_status,
            b_status=b_status,
            c_status=c_status,
            a_full_geometry_status=full_status,
            a_group_sha256s=a_groups,
            b_group_sha256s=b_groups,
            c_group_sha256s=c_groups,
            candidate_a_group_count=a_count,
            candidate_test_group_count=test_count,
            excluded_a_test_overlap_count=a_overlap,
            prior_exposed_group_count=prior_count,
            fresh_test_group_count=fresh_count,
            proposed_b_group_count=b_count,
            proposed_c_group_count=c_count,
            b_quota_on_fresh_groups=b_quota,
            c_quota_on_fresh_groups=c_quota,
        )
        validate_category_proposal(proposal)
        return proposal

    if test_members is None or len(test_members) == 0:
        if category_id in EXPOSED_DEVELOPMENT_CATEGORIES:
            b_status, c_status = "PENDING_AUTHORIZED_B_IDENTITY_RECEIPT", "PENDING_NOT_APPLICABLE_REVIEW"
        elif category_id in PROTECTED_CATEGORIES:
            b_status, c_status = "PENDING_NOT_APPLICABLE_REVIEW", "PENDING_PROTECTED_C_IDENTITY_AUTHORIZATION"
        else:
            b_status = c_status = "PENDING_AUTHORIZED_TEST_IDENTITY_RECEIPT"
        return result("PENDING_AUTHORIZED_IDENTITY_INPUT", b_status=b_status, c_status=c_status,
                      test_count=None if test_members is None else 0,
                      fresh_count=None if test_members is None else 0)

    if not test_identity_scope_verified:
        raise PartitionProtocolError("TEST identity inputs were supplied without a verified L2B identity-only scope")
    if category_id in PROTECTED_CATEGORIES and not protected_test_scope_verified:
        raise PartitionProtocolError("protected TEST identities require separate explicit protected-category scope")
    if not a_groups:
        c_status = "PENDING_NOT_APPLICABLE_REVIEW" if category_id in EXPOSED_DEVELOPMENT_CATEGORIES else "PENDING_A_IDENTITY_AND_DISJOINTNESS"
        return result("PENDING_A_IDENTITY_AND_DISJOINTNESS",
                      b_status="PENDING_A_IDENTITY_AND_DISJOINTNESS", c_status=c_status,
                      test_count=len(test_members))

    test_groups = _member_hashes(test_members, name="TEST population")
    raw_exposed = tuple(prior_exposed_group_sha256s)
    for digest in raw_exposed:
        _require_sha256(digest, "prior-exposure source-group digest")
    if len(set(raw_exposed)) != len(raw_exposed):
        raise PartitionProtocolError("prior-exposure registry contains duplicate group digests")
    if (not exposure_registry_verified
            and category_id not in EXPOSED_DEVELOPMENT_CATEGORIES):
        return result("PENDING_VERIFIED_EXPOSURE_REGISTRY",
                      b_status="PENDING_VERIFIED_EXPOSURE_REGISTRY",
                      c_status=("PENDING_NOT_APPLICABLE_REVIEW" if category_id in EXPOSED_DEVELOPMENT_CATEGORIES
                                else "PENDING_VERIFIED_EXPOSURE_REGISTRY"),
                      test_count=len(test_groups),
                      a_overlap=len(set(a_groups).intersection(test_groups)))
    if not set(raw_exposed).issubset(test_groups):
        raise PartitionProtocolError("prior-exposure registry does not match the authorized TEST source manifest")

    a_overlap = set(a_groups).intersection(test_groups)
    candidates = set(test_groups).difference(a_overlap)
    exposed = set(raw_exposed).intersection(candidates)
    fresh = candidates.difference(exposed)

    if category_id in EXPOSED_DEVELOPMENT_CATEGORIES:
        b_groups = tuple(sorted(candidates))
        c_status = "PENDING_NOT_APPLICABLE_REVIEW"
        if len(b_groups) < BC_MIN_GROUPS_PER_ROLE:
            return result("PENDING_INSUFFICIENT_GROUPS", b_status="PENDING_INSUFFICIENT_GROUPS",
                          c_status=c_status, test_count=len(test_groups), a_overlap=len(a_overlap),
                          prior_count=len(candidates), fresh_count=0, b_count=len(b_groups))
        return result("PROPOSED_REQUIRES_REVIEW",
                      b_status="PROPOSED_EXPOSED_DEVELOPMENT_REQUIRES_REVIEW",
                      c_status=c_status, test_count=len(test_groups), a_overlap=len(a_overlap),
                      prior_count=len(candidates), fresh_count=0, b_count=len(b_groups),
                      b_quota=0, c_quota=0, b_groups=b_groups)

    if category_id in PROTECTED_CATEGORIES:
        if exposed:
            raise PartitionProtocolError("protected C candidate contains previously exposed groups; revise policy before assignment")
        c_groups = tuple(sorted(candidates))
        if len(c_groups) < BC_MIN_GROUPS_PER_ROLE:
            return result("PENDING_INSUFFICIENT_GROUPS", b_status="PENDING_NOT_APPLICABLE_REVIEW",
                          c_status="PENDING_INSUFFICIENT_GROUPS", test_count=len(test_groups),
                          a_overlap=len(a_overlap), fresh_count=len(fresh), c_count=len(c_groups),
                          b_quota=0, c_quota=len(c_groups))
        return result("PROPOSED_REQUIRES_REVIEW", b_status="PENDING_NOT_APPLICABLE_REVIEW",
                      c_status="PROPOSED_PROTECTED_SEALED_REQUIRES_REVIEW",
                      test_count=len(test_groups), a_overlap=len(a_overlap), fresh_count=len(fresh),
                      c_count=len(c_groups), b_quota=0, c_quota=len(c_groups), c_groups=c_groups)

    ranked = sorted(fresh, key=lambda group: _rank_key(category_id, group))
    b_quota = (3 * len(ranked)) // 5
    c_quota = len(ranked) - b_quota
    b_groups = tuple(sorted(exposed.union(ranked[:b_quota])))
    c_groups = tuple(sorted(ranked[b_quota:]))
    if len(b_groups) < BC_MIN_GROUPS_PER_ROLE or len(c_groups) < BC_MIN_GROUPS_PER_ROLE:
        return result("PENDING_INSUFFICIENT_GROUPS", b_status="PENDING_INSUFFICIENT_GROUPS",
                      c_status="PENDING_INSUFFICIENT_GROUPS", test_count=len(test_groups),
                      a_overlap=len(a_overlap), prior_count=len(exposed), fresh_count=len(fresh),
                      b_count=len(b_groups), c_count=len(c_groups), b_quota=b_quota, c_quota=c_quota)
    return result("PROPOSED_REQUIRES_REVIEW", b_status="PROPOSED_REQUIRES_REVIEW",
                  c_status="PROPOSED_REQUIRES_SEALED_REVIEW", test_count=len(test_groups),
                  a_overlap=len(a_overlap), prior_count=len(exposed), fresh_count=len(fresh),
                  b_count=len(b_groups), c_count=len(c_groups), b_quota=b_quota, c_quota=c_quota,
                  b_groups=b_groups, c_groups=c_groups)


def validate_category_proposal(proposal: CategoryPartitionProposal) -> None:
    if proposal.schema != SCHEMA or proposal.category_id not in CANONICAL_CATEGORY_IDS:
        raise PartitionProtocolError("unknown L2E proposal schema or category")
    if proposal.policy_sha256 != POLICY_SHA256 or proposal.seed != PARTITION_SEED:
        raise PartitionProtocolError("proposal is not bound to the frozen L2E policy and seed")
    if proposal.ranking_algorithm != PARTITION_RANKING_ALGORITHM:
        raise PartitionProtocolError("proposal ranking algorithm is incompatible")
    _require_sha256(proposal.source_manifest_sha256, "source-manifest SHA-256")
    for name, groups in (("A", proposal.a_group_sha256s),
                         ("B", proposal.b_group_sha256s),
                         ("C", proposal.c_group_sha256s)):
        for digest in groups:
            _require_sha256(digest, f"{name} group identity")
        if len(groups) != len(set(groups)):
            raise PartitionProtocolError(f"{name} role proposal contains duplicate groups")
    if (set(proposal.a_group_sha256s).intersection(proposal.b_group_sha256s)
            or set(proposal.a_group_sha256s).intersection(proposal.c_group_sha256s)
            or set(proposal.b_group_sha256s).intersection(proposal.c_group_sha256s)):
        raise PartitionProtocolError("A/B/C role proposal contains source-group overlap")
    if (proposal.contains_raw_sample_keys or proposal.contains_image_paths
            or proposal.contains_labels_or_masks):
        raise PartitionProtocolError("proposal exposes prohibited identity or outcome fields")
    expected = canonical_sha256(_proposal_body(proposal))
    if proposal.proposal_sha256 != expected:
        raise PartitionProtocolError("proposal SHA-256 does not match its canonical payload")
    if proposal.status.startswith("CERTIFIED") or any(
        status.startswith("CERTIFIED")
        for status in (proposal.a_status, proposal.b_status, proposal.c_status)
    ):
        raise PartitionProtocolError("a deterministic proposal cannot claim certification")


def validate_global_role_disjointness(
    role_groups_by_category: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    """Reject a group digest assigned more than once across any category/role."""
    seen: dict[str, tuple[str, str]] = {}
    for category_id in sorted(role_groups_by_category):
        if category_id not in CANONICAL_CATEGORY_IDS:
            raise PartitionProtocolError("global partition contains an unknown category")
        role_groups = role_groups_by_category[category_id]
        if set(role_groups) != set(ROLE_NAMES):
            raise PartitionProtocolError("global partition must provide exactly A/B/C per category")
        for role in ROLE_NAMES:
            groups = tuple(role_groups[role])
            for digest in groups:
                _require_sha256(digest, f"{category_id}/{role} group identity")
            if len(groups) != len(set(groups)):
                raise PartitionProtocolError(f"{category_id}/{role} repeats a source group")
            for digest in groups:
                previous = seen.get(digest)
                if previous is not None:
                    raise PartitionProtocolError(
                        f"source group is assigned more than once ({previous[0]}/{previous[1]} and {category_id}/{role})")
                seen[digest] = (category_id, role)


def build_offline_evaluator_stress_order(
    group_sha256s: Sequence[str],
    *,
    order_name: str,
    labels_by_group: Mapping[str, int] | None = None,
    seed: int = TEMPORAL_ORDER_SEED,
) -> tuple[str, ...]:
    """Order one fixed group multiset; labels are evaluator-only inputs.

    The online controller receives only the resulting event order, never this
    label map. RANDOM uses a domain-separated stable hash rank and ignores it.
    """
    if seed != TEMPORAL_ORDER_SEED:
        raise PartitionProtocolError("temporal order seed differs from the frozen L2E proposal")
    if order_name not in {"RANDOM", "NORMAL_FIRST", "DEFECT_FIRST", "BURSTY_50_GROUP"}:
        raise PartitionProtocolError("unknown temporal stress-order profile")
    groups = tuple(group_sha256s)
    for digest in groups:
        _require_sha256(digest, "temporal source-group digest")
    if len(groups) != len(set(groups)):
        raise PartitionProtocolError("temporal stress input contains duplicate groups")
    rank = lambda group: (
        hashlib.sha256(
            f"{TEMPORAL_ORDER_RANKING_ALGORITHM}\0{seed}\0{group}".encode("ascii")
        ).hexdigest(),
        group,
    )
    if order_name == "RANDOM":
        return tuple(sorted(groups, key=rank))
    if labels_by_group is None or set(labels_by_group) != set(groups):
        raise PartitionProtocolError("offline stress order requires one evaluator label per group")
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels_by_group.values()):
        raise PartitionProtocolError("offline stress labels must be binary 0/1 values")
    if order_name == "NORMAL_FIRST":
        return tuple(sorted(groups, key=lambda group: (labels_by_group[group], rank(group))))
    if order_name == "DEFECT_FIRST":
        return tuple(sorted(groups, key=lambda group: (-labels_by_group[group], rank(group))))

    remaining = {
        label: sorted((group for group in groups if labels_by_group[group] == label), key=rank)
        for label in (0, 1)
    }
    ordered: list[str] = []
    current_label = 0
    while remaining[0] or remaining[1]:
        if not remaining[current_label]:
            current_label = 1 - current_label
        ordered.extend(remaining[current_label][:BURSTY_BLOCK_GROUP_COUNT])
        del remaining[current_label][:BURSTY_BLOCK_GROUP_COUNT]
        current_label = 1 - current_label
    return tuple(ordered)


def researcher_approval_binding_errors(
    approval: ResearcherApproval | None,
    *,
    expected_l2b_proposal_sha256: str,
) -> tuple[str, ...]:
    """Check L2E semantics after L2B's signature/trust-root validator succeeds."""
    _require_sha256(expected_l2b_proposal_sha256, "L2B policy proposal SHA-256")
    if approval is None:
        return ("researcher approval is missing",)
    errors: list[str] = []
    if approval.proposal_sha256 != expected_l2b_proposal_sha256:
        errors.append("approval is bound to a different L2B proposal")
    if approval.decision != "APPROVE_H_SOURCE":
        errors.append("approval decision is not APPROVE_H_SOURCE")
    if approval.decision_id != f"R2-L2E-FREEZE:{POLICY_SHA256}":
        errors.append("signed decision_id does not bind the exact L2E policy SHA-256")
    if tuple(approval.approved_categories) != CANONICAL_CATEGORY_IDS:
        errors.append("approval does not cover the complete 30-category public inventory")
    if tuple(approval.approved_roles) != ROLE_NAMES:
        errors.append("approval does not cover A/B/C roles")
    if tuple(approval.approved_protected_policy_categories) != PROTECTED_CATEGORIES:
        errors.append("approval does not bind the proposed protected category treatment")
    return tuple(errors)
