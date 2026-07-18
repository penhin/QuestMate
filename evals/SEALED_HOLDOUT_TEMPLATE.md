# Sealed Holdout Handoff

This file is for the independent evaluation owner. Do not add sealed cases,
answers, URLs, aliases, or per-case reports to this repository.

1. Create a private JSONL dataset with game-family and source-domain splits
   disjoint from public cases.
2. Create its sidecar manifest with `sealed: true`, `refresh_required: false`,
   and a SHA-256 of that dataset.
3. Run only from the controlled environment using `--sealed-holdout`; return
   aggregate pass rate, citation-grounding rate, error count, and latency
   percentiles to implementers.
4. Rotate the holdout after any per-case result, source URL, or answer is
   revealed to an implementer.
