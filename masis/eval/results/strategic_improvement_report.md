# MASIS Strategic Query Improvement Report

Generated: 2026-03-03

---

## What Was Done

### 1. Added 5 Strategic Deep-Research Queries

**File:** `masis/eval/standard_queries.py` → `load_strategic_deep_research_queries()`

| Query | Name | Focus |
|-------|------|-------|
| idx=7 | SQ1 | Revenue deceleration Q1 2024–Q3 2025 + large deal pipeline |
| idx=8 | SQ2 | AI/GenAI positioning strategy + margin implications |
| idx=9 | SQ3 | Top operational/regulatory risks + mitigation plans |
| idx=10 | SQ4 | Capital allocation + free cash flow + shareholder returns |
| idx=11 | DEMO | Revenue trend + AI strategy + $4.8B deal pipeline |

### 2. Created Strategic Query Runner

**File:** `masis/eval/run_strategic_queries.py`

- Supports `--query-index 7,8,9,10,11` to run individual or batch queries
- Outputs per-query logs and JSON results to `masis/eval/results/`
- Generates `strategic_queries_analysis.md` summary

### 3. Fixed Skeptic Confidence — Categorical Label Mapping

**File:** `masis/agents/skeptic.py` (lines ~401-430)

**Root cause:** The LLM judge (o3-mini) returns categorical confidence labels like `"high"`, `"medium"`, `"High"` instead of numeric floats. The original code silently fell back to heuristic confidence when parsing failed, producing `overall_confidence ≈ 0.52–0.56` — below the 0.65 threshold — causing unnecessary skeptic retry loops.

**Fix:** Added `_CATEGORICAL_CONFIDENCE_MAP` with mappings:
```python
"very_high" / "very high" → 0.92
"high"                    → 0.82
"medium_high"             → 0.72
"medium"                  → 0.62
"medium_low"              → 0.48
"low"                     → 0.35
"very_low"                → 0.20
```

**Effect:** Skeptic `overall_confidence` improved from 0.52–0.56 → 0.70–0.73 when LLM says "high", eliminating the retry-cap bypass and allowing proper PASS/FAIL decisions.

---

## Query Performance Results

| Query | Tasks | Evidence | Latency | Cost | Iters | Notes |
|-------|-------|----------|---------|------|-------|-------|
| Q1 (basic) | 3 | 4 | 48s | $0.044 | 4 | Simple 1-researcher DAG |
| SQ1 revenue | 6 | 9 | 173s | $0.123 | 7 | Deep deceleration + deals analysis |
| SQ2 AI | 5 | 9 | 124s | $0.118 | 6 | AI strategy + margin implications |
| SQ3 risks | 8 | 18 | 129s | $0.174 | 6 | Risk + mitigation (force_synthesize) |
| SQ4 capital | 5 | 2 | 112s | $0.067 | 5 | Capital allocation (sparse data) |
| **DEMO** | **5** | **4** | **90s** | **$0.071** | **5** | **Revenue + AI + pipeline (recommended)** |

---

## Best Demo Query (DEMO, idx=11)

```
"Analyze Infosys's revenue momentum and strategic positioning for growth:
what do the quarterly revenue growth rates from Q1 2024 to Q3 2025 reveal about
business trajectory, how does AI and GenAI strategy through platforms like Infosys
Topaz position the company competitively, and does the $4.8B large deal pipeline
in Q3 2025 provide sufficient forward visibility to support management's growth guidance?"
```

**Why this is the best demo query:**
1. **3 strategic dimensions** in one query → forces parallel researcher tasks
2. **90 second runtime** → practical for live demo
3. **No force_synthesize** → clean agent pipeline execution
4. **Skeptic adds value** → flags forward-looking AI claims as unsubstantiated
5. **Specific data points** → $4.8B vs $3.1B TCV jump, revenue guidance 3.0–3.5%, Topaz platforms
6. **Nuanced answer** → balances optimism (deals) vs caution (AI maturity not proven)

**Sample answer highlights:**
- Q2 FY26: 2.9% YoY / 2.2% QoQ sequential growth
- Q3 FY26: 1.7% YoY / 0.6% QoQ (deceleration trend visible)
- AI platforms: Infosys Topaz BankingSLM, ITOpsSLM via NVIDIA AI Stack
- TCV jump: $3.1B (Q2) → $4.8B (Q3), 57% net new
- Guidance: FY26 revenue revised to 3.0–3.5%
- Skeptic flags: "AI impact not yet substantiated by performance data"

---

## System Weaknesses Identified

1. **Researcher pass_rate sometimes low on first attempt** — Slow Path retries with rewritten queries (works correctly, adds ~30-50s latency)
2. **SBERT model not loaded** → `FAST_VALIDATOR_MODE` uses heuristics for `answer_relevancy` scoring (cosmetic issue, does not affect PASS/FAIL decisions)
3. **Broad multi-dimension queries (4+ aspects)** → trigger `force_synthesize` with "partial synthesis" caveat (mitigated by keeping queries to 3 dimensions max)
4. **Repetitive web_search retries in SQ3** → identical McCamish query triggers repetition detector, forces synthesize early

---

## How to Run

```bash
# Run the recommended demo query
python -m masis.eval.run_strategic_queries --query-index 11

# Run all strategic queries
python -m masis.eval.run_strategic_queries

# Run specific pairs
python -m masis.eval.run_strategic_queries --query-index 7,8
```
