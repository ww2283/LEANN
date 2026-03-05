# Changelog

All notable changes to LEANN are documented here. Append-only, newest entries at the bottom.

Format: `## YYYY-MM-DD: <short summary>` followed by bullet points.

## 2026-03-05: IVF backend incremental update support

- Added `leann-backend-ivf` with FAISS IndexIVFFlat + DirectMap.Hashtable.
- IVF supports in-place `add_vectors` and `remove_ids` without full rebuild.
- `leann build` is now idempotent: re-running on an existing index does incremental update (add new, remove deleted, re-index modified files).
- Fixed incremental build chunking inconsistency and shared metadata dict bug.
- Fixed IVF incremental update duplicate chunks from stale `passages.jsonl`.

## 2026-03-05: MCP server v2 — build, status, and structured search

- Added `leann_build` MCP tool: build or incrementally update indexes directly from Claude Code.
- Added `leann_status` MCP tool: inspect index details (backend, embedding model, chunk/file count, size).
- `leann_search` now uses `--json` output with file paths always included, formatted as markdown code blocks.
- Fixed `float32` JSON serialization bug in `leann search --json`.
- Cleaned up MCP tool descriptions (concise, no emoji).

## 2026-03-05: Documentation — roadmap, vision, and dev guidelines

- Rewrote `docs/roadmap.md` with current P0/P1 priorities from GitHub issue #237.
- Added `docs/ultimate_goal.md` — long-term vision (personal data platform, best code retrieval MCP, multimodal, local-first).
- Added self-contained documentation principle and dev doc maintenance rules to `CLAUDE.md`.
