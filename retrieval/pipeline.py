"""Stage-oriented evidence fusion for retrieval results.

Retrievers intentionally return the public ``Source`` model so callers do not
need to know whether a result came from a local chunk, MediaWiki, or web
search.  This module adds the internal, passage-level layer between retrieval
and the answer-facing source pool.  It preserves provenance long enough to
make fusion observable without expanding the public API contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from quality_policy import source_domain_limit
from retrieval.evidence_pool import canonical_source_url, merge_source_evidence, source_rank
from schemas import Source


@dataclass(frozen=True)
class EvidenceCandidate:
    """One answerable passage returned by a retrieval channel."""

    source: Source
    channel: str
    passage: str
    rank: float


@dataclass(frozen=True)
class RetrievalStage:
    """A bounded, request-local record of one retrieval pipeline stage."""

    name: str
    input_count: int
    output_count: int
    details: dict[str, int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedEvidencePool:
    """Answer-facing sources plus stage counts suitable for structured logs."""

    sources: list[Source]
    candidate_count: int
    fused_count: int
    channels: dict[str, int]
    stages: list[RetrievalStage]


def fuse_and_rank_evidence(
    *,
    groups: dict[str, list[Source]],
    query: str,
    intent: str,
    max_results: int,
    version_sensitive: bool = False,
    entity_groups: list[list[str]] | None = None,
) -> FusedEvidencePool:
    """Fuse duplicate pages at passage granularity before final source selection.

    A single document may be returned by a local chunk index and live search,
    or by several complementary chunks.  Ranking each passage first avoids a
    broad high-score excerpt masking a later direct answer from that same URL.
    """
    channels = {channel: len(sources) for channel, sources in groups.items()}
    candidates: list[EvidenceCandidate] = []
    for channel, sources in groups.items():
        for source in sources:
            passage = (source.evidence or source.snippet or "").strip()
            passage_source = source.model_copy(update={"evidence": passage or None})
            candidates.append(
                EvidenceCandidate(
                    source=passage_source,
                    channel=channel,
                    passage=passage,
                    rank=source_rank(
                        source=passage_source,
                        query=query,
                        intent=intent,
                        version_sensitive=version_sensitive,
                        entity_groups=entity_groups,
                    ),
                )
            )

    by_url: dict[str, list[EvidenceCandidate]] = {}
    for candidate in candidates:
        by_url.setdefault(canonical_source_url(str(candidate.source.url)), []).append(candidate)

    fused: list[tuple[float, Source]] = []
    for candidates_for_url in by_url.values():
        ordered = sorted(candidates_for_url, key=lambda candidate: candidate.rank, reverse=True)
        merged = ordered[0].source
        for candidate in ordered[1:]:
            merged = merge_source_evidence(preferred=merged, other=candidate.source)
        # Keep the strongest directly relevant passage as the ordering signal;
        # the merged text is retained for claim extraction and citations.
        fused.append((ordered[0].rank, merged))

    selected: list[Source] = []
    domain_counts: dict[str, int] = {}
    for _rank, source in sorted(fused, key=lambda item: item[0], reverse=True):
        domain = urlparse(str(source.url)).netloc.casefold()
        if domain_counts.get(domain, 0) >= source_domain_limit(domain):
            continue
        selected.append(source)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= max_results:
            break

    return FusedEvidencePool(
        sources=selected,
        candidate_count=len(candidates),
        fused_count=len(fused),
        channels=channels,
        stages=[
            RetrievalStage(
                name="candidate_build",
                input_count=sum(channels.values()),
                output_count=len(candidates),
                details={"channels": len(channels)},
            ),
            RetrievalStage(
                name="passage_fusion",
                input_count=len(candidates),
                output_count=len(fused),
                details={"deduplicated": len(candidates) - len(fused)},
            ),
            RetrievalStage(
                name="rerank_and_select",
                input_count=len(fused),
                output_count=len(selected),
                details={"max_results": max_results},
            ),
        ],
    )
