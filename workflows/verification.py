"""Deterministic evidence checkpoint selection inside task workflows."""

from enum import StrEnum

from schemas import SearchPlan


class EvidencePath(StrEnum):
    RESEARCH = "evidence_research"
    VERIFIED_RESEARCH = "verified_research"


class EvidenceVerificationRouter:
    @staticmethod
    def classify(plan: SearchPlan) -> EvidencePath:
        if plan.version_sensitive or plan.requires_relation_verification or len(plan.named_entity_groups) >= 2:
            return EvidencePath.VERIFIED_RESEARCH
        return EvidencePath.RESEARCH
