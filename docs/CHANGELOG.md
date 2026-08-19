# Changelog

Append-only log of major changes to LEANN (new features, breaking changes, important
fixes). Newest entries at the bottom.

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

## 2026-06-02: GPU FlashLib IVF backend (`flashlib_ivf`)

- Add `leann-backend-flashlib-ivf`, a GPU IVF-Flat (inverted file) approximate-NN
  backend built on FlashLib (`flash_ivf_flat`, Triton/CuteDSL) — the GPU counterpart
  of the FAISS `ivf` backend. Registered as backend name `flashlib_ivf`; install via
  `uv sync --extra flashlib-ivf` or `pip install leann-backend-flashlib-ivf`. Shares
  the `nlist`/`nprobe` recall knobs with the `ivf` backend, so the two are drop-in
  comparable. Requires a CUDA GPU at build (k-means) and search.
- Add `benchmarks/flashlib_ivf_vs_faiss_ivf.py`: head-to-head `flashlib_ivf` (GPU) vs
  `ivf` (FAISS, CPU) at matched `nlist` across an `nprobe` sweep (build time,
  single-query latency, batched throughput, recall@k vs exact ground truth). On an
  NVIDIA H200 at 1M x 768 vectors (nlist=4096, 8 CPU threads): ~13x faster build and,
  at nprobe=32, ~6.5x lower single-query latency / ~75x higher batched throughput at
  comparable recall (GPU latency stays ~flat while CPU grows linearly with nprobe).
- Docs: `docs/flashlib_backend_guide.md` gains a `flashlib_ivf` section.

## 2026-08-19: Multi-invocation incremental builds (`--sync-key`, `changes`, `verify`) + review hardening

- `leann build --sync-key <key>`: stable snapshot identity shared across invocations with
  different `--docs` lists (single keyed Merkle snapshot instead of per-root snapshots).
  Mismatched keys on a keyed index are rejected unless `--force` rekeys.
- New `leann changes` subcommand: non-mutating JSON report of pending added/modified/removed
  files vs the stored snapshot. Errors (exit 1) on a missing index, empty sync scope, wrong
  sync key, or corrupt snapshot instead of reporting a false clean delta.
- New `leann verify` subcommand: cross-artifact integrity check (meta.json, passages.jsonl,
  passages.idx offsets, IVF id map inversion, FAISS vector count).
- Safely-empty incremental deltas (zero new chunks, nothing modified/removed) now commit the
  snapshot so re-runs report "up to date" — but only when no document failed to load; a
  swallowed loader failure aborts the build without committing, so the failed files stay pending.
- Corrupt sync snapshots and unreadable `sync_roots.json` now fail loud on build (recover
  with `--force`) instead of silently degrading to full-rediff or unkeyed identity.
- A transiently unreadable file keeps its previous hash instead of being classified as
  removed (which deleted its chunks from the index).
- `LEANN_NO_REGISTER=1` env switch skips project-directory registration (for tests/CI).

## 2026-08-19: verify accepts duplicate content-hash passage ids (issue #5)

- `leann verify` no longer false-positives on healthy indexes built with
  `--id-scheme=content-hash`, where byte-identical chunks legitimately share one passage id
  (many-to-one by construction: jsonl keeps one line per chunk, the offset map and IVF
  `passage_to_id` are last-wins one-entry-per-unique-id).
- Invariants rewritten to the actual contract: duplicate jsonl ids allowed only with
  identical text per id; idx cardinality vs unique ids; IVF `passage_to_id` checked as a
  partial inverse (each passage id maps back to one of its FAISS labels, key set equals
  the deduped `id_to_passage` value set).
- Same id with different text, broken inversion, or a missing `passage_to_id` entry still
  fail verify; tests now include a duplicate-content fixture.

## 2026-08-19: incremental add gave all chunks of a file one passage id (issue #7)

- Chunkers can return chunks for one file that alias a single shared metadata dict; both
  id assigners in `cli.py` (`_assign_chunk_ids`, `_assign_unique_chunk_ids`) wrote each
  per-chunk id into that shared dict, so the last chunk's id overwrote all previous ones
  and `LeannBuilder.add_text` (which resolves the passage id from `metadata["id"]`) gave
  every chunk of a multi-chunk file the same id — corrupting `passages.idx` and the FAISS
  id map, and failing `leann verify` on every multi-file incremental build.
- Fixed by copying the metadata dict at assignment time
  (`c["metadata"] = {**c.get("metadata", {}), "id": sid}`); regression tests cover both
  assigners with a shared-dict fixture.
