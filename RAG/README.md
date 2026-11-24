# ReliVE Surgical Visual Knowledge Base V0.1

This is an **offline, benchmark-independent reference-corpus builder**. It does
not answer a QA sample, inspect surgical videos, retrieve knowledge, create a
vector database, run an agent or alter ReliVE inference/certificates. Existing
`interview_sop_public.json` and reference PDFs are read-only inputs, not training
examples or a source of extra clinical facts for this builder.

## Current local delivery

The initial full inspection found 46 PDFs: 17 at the top level and 29 in a
recursively discovered subdirectory, totaling 2,136 PDF pages. All documents
opened and all pages were parsed; 130 pages were classified as image/sparse
text and remain pending fuller visual interpretation. A total of 1,094 pages
have image/table/sparse-text signals; this is a candidate count, not a claim
that all require a paid vision call. Two nursing books contain 1,389 and 421
pages; they are filtered locally before any extraction request.

No API was configured locally. The formal filename exists with
`build_status=INSPECT_ONLY`, a complete source index and **zero knowledge cards**.
All six production category counts are zero. Real API smoke and full semantic
extraction have not been executed. Do not mistake inspect success for a finished
medical corpus. The separately named mock smoke produced eight extractive
fixtures, all flagged synthetic, and is forbidden from the formal library.

## Install and run

Run from the repository root with Python 3.10+ on Linux/macOS. Use a dedicated
environment; this does not require the GPU/PyTorch/Transformers environment.

```bash
python -m venv RAG/.venv
source RAG/.venv/bin/activate
python -m pip install -r RAG/requirements.txt

# Inspect all PDFs, recursively. No API key or network is needed.
python -m RAG.knowledge_builder.build --mode inspect \
  --input-dir RAG/references \
  --cache-dir RAG/cache \
  --output RAG/output/inspection/relive_surgical_visual_knowledge_v0.1.json \
  --verbose

# --dry-run is an alias for inspect even if --mode build is specified.
# An unchanged completed inspection can be verified/reused without API calls.
python -m RAG.knowledge_builder.build --mode inspect --resume \
  --input-dir RAG/references --cache-dir RAG/cache \
  --output RAG/output/inspection/relive_surgical_visual_knowledge_v0.1.json

# The three required source categories, deliberately isolated mock output.
python -m RAG.knowledge_builder.build --mode mock \
  --input-dir RAG/references --cache-dir RAG/cache \
  --include-file common_surgical_instruments_module.pdf \
  --include-file DualKnife_J_Professor_Naohisa_Yahagi.pdf \
  --include-file OpenSurgery_pdfs/003_handling-of-surgical-instruments.pdf \
  --max-pages 5 --output RAG/output/mock_smoke/knowledge.mock.json

python -m unittest discover -s RAG/tests -v
```

`--max-files` and `--max-pages` limit processing, not discovery. Every discovered
PDF remains indexed; skipped files/pages are explicitly deferred. `--include-file`
is repeatable and uses exact paths relative to references. No folder name or PDF
count is hard-coded. Unicode/space paths, `.PDF`, parse errors and symlink escape
attempts are handled explicitly. Symlink directories are reported, not followed.
`--no-tables` disables vector-table extraction and reports it as NOT_REQUESTED.
Document deduplication uses identical file bytes, or matching full parsed text
plus rendered pages. Equal text with different diagrams is reported, not skipped;
partial-text matches never establish whole-document duplication.

## Source independence review

Inspection generates `source_manifest.json`, `source_inventory.md`,
`source_audit_report.json` and `source_review_template.jsonl`. Every PDF has a
relative path, page count, SHA-256, extractability, probable source, category,
status and any failure/deferred reason. Metadata is not assumed authoritative;
`title_basis` distinguishes PDF metadata from an extracted candidate title.
Publisher, canonical URL, publication year and license stay null/unknown unless
actually established. A creation/printing date is not a publication year.

Before a **real** build, save a completed copy of the source-review JSONL. Each
row is bound to source ID, path, title and file SHA. Fill only:

* `decision`: INCLUDE / EXCLUDE / PENDING;
* `reviewer_id`, `rationale`, `independence_basis` for INCLUDE;
* `allowed_pages`: [] for a reviewed whole independent source, or a list of
  1-based PDF pages for an explicitly reviewed subset.

No source decision is auto-filled. A review can contain a subset of sources;
omitted sources remain pending. This is a provenance/independence review, not
medical approval of the later generated cards. Unreviewed sources cannot be
sent to the real extractor.

The mixed OpenSurgicalVision document is quarantined by default because it
contains project-specific CVAT/mask material. Any attempted admission requires
an explicit page allowlist plus reviewer rationale. Even then, benchmark-specific
blocks and output cards are removed. If generic portions cannot be separated
reliably, EXCLUDE the document. Other benchmark/data/answer signatures are
screened in every source block and **every output field**. Generic terms such
as clipping, suture and CVS are allowed. Signature screening is not a proof of
independence; the source review remains necessary.

## Real API smoke on a server

The reference PDFs are now tracked with Git LFS. After fetching this branch,
run `git lfs pull` and verify the source hashes before parsing; a plain LFS
pointer is not a PDF. The initial inspect/mock audit snapshots and SOP JSON
are also tracked. Cache, API keys, environments and future output directories
remain ignored. See [the Chinese server runbook](SERVER_RUNBOOK.zh-CN.md).
Paths are CLI inputs, never fixed to a developer/server filesystem.

Configure an OpenAI-compatible **Chat Completions** endpoint. `LLM_BASE_URL`
includes the API prefix such as `/v1`, but excludes `/chat/completions`. No
provider, base URL or model is silently selected. The key is read only from
`LLM_API_KEY` (or a caller-configured credential environment variable in Python).
The example env file contains no secrets and is **not automatically loaded**.
Use your server's secret injection, or a locally ignored env file you source.

`LLM_ENABLE_THINKING` is an optional provider extension. Leave it empty/unset
to omit the request parameter, or set exactly `true`/`false` (case-insensitive)
to send a JSON boolean. Invalid values stop configuration. For Qwen4-Max,
the API model ID is `qwen4-max`; set `LLM_ENABLE_THINKING=false` to reproduce
the non-thinking connectivity probe during structured extraction. See the
[provider's structured-output documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output).
This setting is part of the cache/build identity. After changing it, use a
new output directory; historical failures remain unchanged. A successful
connectivity probe does not validate PDF extraction or medical correctness.

```bash
# Edit an ignored local copy; never paste a key into a shared log or git file.
cp RAG/.env.example RAG/.env
# Fill LLM_BASE_URL, LLM_TEXT_MODEL and LLM_API_KEY using secure local tooling.
set -a
source RAG/.env
set +a

PDF_ROOT="$PWD/RAG/references"
CACHE_ROOT="$PWD/RAG/cache"
SOURCE_REVIEW="$PWD/RAG/source_review.completed.jsonl"
# First copy inspection/source_review_template.jsonl to SOURCE_REVIEW and
# complete the source independence decisions described above.
SMOKE_OUT="$PWD/RAG/output/api_smoke_$(date +%Y%m%d_%H%M%S)"

python -m RAG.knowledge_builder.build --mode build \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --source-review "$SOURCE_REVIEW" \
  --include-file common_surgical_instruments_module.pdf \
  --include-file DualKnife_J_Professor_Naohisa_Yahagi.pdf \
  --include-file OpenSurgery_pdfs/003_handling-of-surgical-instruments.pdf \
  --max-pages 5 --max-api-calls 32 \
  --output "$SMOKE_OUT/relive_surgical_visual_knowledge_v0.1.json" --verbose

# No API calls: reopen the real files, verify hashes/pages and literal quotes.
python -m RAG.knowledge_builder.build --mode validate \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --output "$SMOKE_OUT/relive_surgical_visual_knowledge_v0.1.json"
```

Inspect source quotes and the generated facts/features/guidance; confirm no
unlabeled figure inference, commercial performance claim or unsafe sufficiency
rule slipped through. All cards must stay pending medical review. Only after
the real smoke passes should the full server build run:

```bash
FULL_OUT="$PWD/RAG/output/full_$(date +%Y%m%d_%H%M%S)"
python -m RAG.knowledge_builder.build --mode build \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --source-review "$SOURCE_REVIEW" --max-api-calls 2000 \
  --output "$FULL_OUT/relive_surgical_visual_knowledge_v0.1.json" --verbose

python -m RAG.knowledge_builder.build --mode validate \
  --input-dir "$PDF_ROOT" --cache-dir "$CACHE_ROOT" \
  --output "$FULL_OUT/relive_surgical_visual_knowledge_v0.1.json"
```

These are API-backed extractions, not GPU experiments. The call ceiling includes
retries and capability fallbacks. Source-consistency checks add a second logical
request for each nonempty extraction unit. A full run may need a different budget
after inspecting candidate-unit counts; do not interpret 2,000 as a guaranteed
complete/exhaustive extraction. No API call occurs in inspect/mock/validate.

## Optional vision and cost control

Vision is off by default. Set `LLM_VISION_MODEL` and use `--enable-vision
--max-vision-pages 3`. Only selected, source-approved pages with admissible labeled
context are rendered and sent. Image-only/unlabeled pages stay pending rather
than receiving a guessed anatomical or instrument identity. Rendered figures
use `extraction_method=vision_llm`, `basis=figure_interpretation`, explicit label
and PDF page. A figure number alone is insufficient. A model accepting an image
request does not independently prove it used the pixels correctly.

Configuration supports timeout, retries, concurrency, requests/minute, output
format and optional per-million-token prices; see `.env.example` and
`provider.Config`. Concurrency-sized batches plus a shared semaphore and rate
limiter prevent unlimited dispatch. Unsupported structured output negotiates
json_schema -> json_object -> schema-in-prompt **only on recognized capability
errors**, always followed by local JSON Schema validation. Unsupported vision
fails explicitly, with no text-only impersonation. Failed raw model responses
are retained; HTTP error bodies/headers and credentials are not logged.

`LLM_MAX_OUTPUT_TOKENS` configures the positive integer output ceiling (default
6,000); it is included in request/cache/build identity and cost reservation.
The Qwen4 smoke's six failed responses all reached 6,000 completion tokens and
ended inside JSON. For a new bounded smoke, explicitly set 24,000 and keep the
same prompts, source selection and validation rules. This raises the per-request
token ceiling fourfold, not a guarantee of completion or an instruction to fill
the allowance. Do not overwrite or resume a completed failed build.

Adapter v0.2 records the allowlisted provider `finish_reason`, request identity,
output ceiling and JSON syntax location/schema path. `finish_reason=length`
is rejected as `MODEL_OUTPUT_TRUNCATED` even if the prefix parses. Invalid JSON
at the token ceiling without a finish reason is conservatively classified the
same way, with `truncation_evidence=INVALID_JSON_AT_TOKEN_LIMIT`. Such a response
is retained and stops that request after one attempt; no identical-limit retry,
partial-card salvage or automatic budget escalation occurs. Ordinary JSON and
schema errors retain bounded retries and distinct reasons. Old response records
did not capture termination evidence; their absence is not retrospectively
filled in. New parser admission uses a new cache namespace; old cache stays
untouched. `attempt_failure_reason_counts` includes failed attempts even when a
later attempt recovers. Existing exact-quote and source-consistency checks apply.

`--max-cost-usd` requires explicit input/output token rates. It reserves a
conservative text-token allowance before each request. This is an estimated
guard for a compliant provider, not an independent billing guarantee. When any
usage is missing, the total estimated cost is null and known usage is separate.
Vision cannot be combined with this hard text-cost reservation because image
pricing is provider-specific; use the API-call/page ceiling instead. Latency,
token use, missing usage, capabilities and configured prices are audited.

## Formats and quality boundary

Six card types: instrument, anatomy, action, interaction, procedure,
evidence_criteria. `schemas/extraction.schema.json` is the closed model response;
`schemas/library.schema.json` validates the persisted JSON. Empty unsupported
feature arrays are valid. Each factual feature and summary has source-fact IDs.
Every source fact carries original-language quote, source ID, PDF page and block
ID. Derived guidance has its own fact dependencies and derivation rationale;
guidance items explicitly link those derivations. No guessed times, video ROI,
source URL, procedure occurrence or truth/certificate status is generated.

The deterministic validator checks quote containment in the actual supplied
block, exact fact-page/reference coverage, figure-label context, source review,
IDs and forbidden data fields. A second schema-bound LLM pass checks source
consistency. Unsupported/uncertain findings are withheld in
`excluded_knowledge.jsonl`. **Neither exact quotation nor a second LLM check proves
medical correctness or semantic entailment.** All surviving cards remain pending;
there is no approved/default trusted retrieval collection or retrieval API.
Future consumers must implement their own medical-review admission.

Normalization preserves model variants, facts and source references. Exact
matching semantic fields merge provenance; aliases differing between models
are kept separate. Same-entity paraphrases/synonyms enter a near-duplicate queue;
they are not automatically merged using a similarity threshold. Possible polarity
conflicts are conservatively withheld. No claim of complete clinical conflict
detection or semantic deduplication is made.

PyMuPDF text blocks, fonts, page geometry and table candidates retain provenance.
Two-column ordering is a heuristic flagged for review, with original blocks
preserved. Repeated margin text, navigation, cookie lines and standalone URLs
have an explicit removal audit. Vector-line table extraction does not establish
that all raster/borderless tables were recovered. OCR is not automatically run;
missing/sparse/image-dependent content is exposed in the page audit. Printed
web pages can contain only video thumbnails, which do not supply the video's
motion evidence. See the [PyMuPDF text/block documentation](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).

## Resume, artifacts and limitations

Parsed documents and every successful extraction/consistency response are
content-addressed. Keys bind PDF hash, unit/page range, prompt/schema, model,
base URL and input/image content. Secret values are never part of a key or log.
Parse-cache records have integrity hashes. Each HTTP attempt has a separate
audit; only validated successful responses are reused. Resume after an interrupted
process uses the same output/config identity and existing caches. A completed
run is immutable: `--resume` verifies/reuses it without new calls. For a finalized
budget-stopped/failed build, use a **new output directory with the same cache**
to continue; this preserves the previous failure report. Changing prompt/model,
inputs, source review or code also requires a new output directory.

Output files include the main library, complete source manifest/inventory,
source-review template, source audit, page parsing/selection audits, accepted
candidate consistency audit, rejected cards, extraction failures, quality report
and hash-bound build summary. `source_excerpt_examples.json` contains literal
inspection excerpts only, **not generated or medically approved cards**.

Exit 0: inspect/mock completed or structurally valid build. Exit 2: API setup,
source-review gate, budget stop, extraction failure or integrity error. A valid
library may be empty or partial; always read `build_status`, failures and source
coverage. `BUILT_PENDING_MEDICAL_REVIEW` describes reviewed-source extraction
completion, not exhaustiveness, clinical approval or evidence verification.

The local inspection proves parsing coverage, not final knowledge coverage.
Real text/vision API capability, extraction quality and billing must still be
tested on the configured server. No R0/R1 artifacts, historical caches, pilot
tags or ReliVE runtime APIs are changed.
