# MASIS Reasoning Simulation

> **Purpose:** Validate the 3-node architecture by tracing queries step-by-step through the system.
> **Architecture:** Supervisor -> Executor -> Validator. Task DAG = data in state.
> **Last Updated:** 2026-03-02

---

## How to Read This Document

Each scenario in this document traces a query **step-by-step** through the system:
- **SUPERVISOR** entries show the routing decision (PLAN / FAST / SLOW)
- **EXECUTOR** entries show which agent runs and what it returns
- **VALIDATOR** entries show quality scores and pass/fail
- **HITL** entries show `interrupt()` -> pause -> resume flow
- **Cost & Latency** are calculated for each scenario

The scenarios are ordered from simple -> complex and cover **every** architectural feature.

Labeling convention in this document:
- **Actual Output (from logs/results)**: copied from `masis/eval/results/*_result.json`.
- **Illustrative Flow (for explanation)**: storytelling examples for presentation.

## Actual Output Snapshot (Recent Artifacts)

Source files:
- `masis/eval/results/infosys_q1_result.json`
- `masis/eval/results/infosys_q2_result.json`
- `masis/eval/results/infosys_q3_result.json`

```json
[
  {
    "scenario": "Q1",
    "latency_s": 190.94,
    "iteration_count": 6,
    "validation_round": 1,
    "supervisor_decision": "ready_for_validation",
    "evidence_count": 10
  },
  {
    "scenario": "Q2",
    "latency_s": 318.53,
    "iteration_count": 7,
    "validation_round": 1,
    "supervisor_decision": "ready_for_validation",
    "evidence_count": 7
  },
  {
    "scenario": "Q3",
    "latency_s": 193.14,
    "iteration_count": 5,
    "validation_round": 1,
    "supervisor_decision": "ready_for_validation",
    "evidence_count": 5
  }
]
```

Q2 shows why watchdog and retry control are required for stable demos.

---


## Scenario 1: Simple Factual Query

### Query: "What was Infosys's Q3 FY26 revenue?"

**Why this scenario matters:** Tests the happy path — single research task, all quality gates pass, Fast Path handles all routing.

```
Step 1 — SUPERVISOR (MODE 1: PLAN — Slow Path, gpt-4.1)
  Input:  "What was Infosys's Q3 FY26 revenue?"
  Action: gpt-4.1 decomposes query → generates TaskPlan with structured output
  Output: TaskPlan(
    tasks=[
      T1(researcher, "Infosys Q3 FY26 revenue",
         criteria: "≥2 chunks, pass_rate≥0.30, self_rag=grounded"),
      T2(skeptic, "Verify revenue claims from T1",
         criteria: "0 unsupported, 0 contradicted, confidence≥0.65"),
      T3(synthesizer, "Synthesize Q3 revenue answer",
         criteria: "all claims cited, all citations valid"),
    ],
    stop_condition="Q3 FY26 revenue figure with source citation"
  )
  Decision: supervisor_decision = "continue", next_tasks = [T1]
  Cost: ~$0.012 (gpt-4.1, 1 call)

Step 2 — EXECUTOR → T1 (Researcher / Eng 6 Pipeline)
  HyDE rewrite: "Infosys posted Q3 FY26 revenue of approximately..."
  Metadata: {year: 2026, quarter: "Q3"}
  Hybrid retrieval: 10 chunks (5 vector + 5 BM25, RRF k=60)
  Cross-encoder rerank: top 5
  CRAG grading: 3/5 relevant (pass_rate=0.60)
  Generate: "Q3 FY26 revenue was ₹41,764 crore, up 12% YoY"
  Self-RAG: "grounded" ✅
  Output: ResearcherOutput(
    evidence=[chunk_12, chunk_24, chunk_38],
    summary="Q3 FY26 revenue ₹41,764 crore, 12% YoY growth",
    chunks_after_grading=3, grading_pass_rate=0.60,
    self_rag_verdict="grounded", source_diversity=2
  )

Step 3 — SUPERVISOR (MODE 2: FAST PATH — No LLM, $0)
  Reads T1.acceptance_criteria: "≥2 chunks, pass_rate≥0.30, self_rag=grounded"
  Checks: chunks=3≥2 ✅  pass_rate=0.60≥0.30 ✅  self_rag="grounded" ✅
  Next ready task: T2 (dependencies: [T1] → T1=done → T2 ready)
  Decision: "continue", next_tasks = [T2]

Step 4 — EXECUTOR → T2 (Skeptic)
  Stage 1 — NLI (BART-MNLI, local, free):
    "₹41,764 crore" vs chunk_12 → ENTAILMENT (0.95)
    "12% YoY" vs chunk_24 → ENTAILMENT (0.88)
    "up from Q3 FY25" vs chunk_38 → ENTAILMENT (0.91)
  Stage 2 — o3-mini judge: no flags, confidence 0.92
  Output: SkepticOutput(
    claims_checked=3, claims_supported=3, claims_unsupported=0,
    claims_contradicted=0, logical_gaps=[], overall_confidence=0.92
  )

Step 5 — SUPERVISOR (MODE 2: FAST PATH — No LLM, $0)
  Reads T2.acceptance_criteria: "0 unsupported, 0 contradicted, confidence≥0.65"
  Checks: unsupported=0 ✅  contradicted=0 ✅  confidence=0.92≥0.65 ✅
  Next ready task: T3
  Decision: "continue", next_tasks = [T3]

Step 6 — EXECUTOR → T3 (Synthesizer, gpt-4.1)
  Evidence: chunk_12 + chunk_24 + chunk_38 (U-shape ordered)
  Critique: none
  Output: SynthesizerOutput(
    answer="Infosys's Q3 FY26 consolidated revenue was ₹41,764 crore, 
    representing 12% year-over-year growth [Annual Report 2026, p.42].",
    citations=[Citation(chunk_id="c12", claim_text="₹41,764 crore", entailment=0.95)],
    claims_count=2, citations_count=2, all_citations_in_evidence_board=True
  )

Step 7 — SUPERVISOR (MODE 2: FAST PATH — No LLM, $0)
  All tasks done (T1=done, T2=done, T3=done)
  Decision: "ready_for_validation"

Step 8 — VALIDATOR
  faithfulness: 0.94 ✅ (threshold: 0.85)
  citation_accuracy: 0.95 ✅ (threshold: 0.90)
  answer_relevancy: 0.97 ✅ (threshold: 0.80)
  dag_completeness: 1.00 ✅ (threshold: 0.90)
  → route_validator: "pass" → END

TOTALS:
  Supervisor LLM calls: 1 (plan only)
  Supervisor Fast Path calls: 4 (free)
  Agent LLM calls: researcher(5) + skeptic(1) + synthesizer(1) = 7
  Cost: ~$0.035
  Latency: ~8s
```

---

## Scenario 2: Multi-Step with Failure and DAG Modification

### Query: "Compare Infosys's cloud revenue to Adobe, and GCP"

**Why this scenario matters:** Tests parallel execution (Send), task failure (internal docs lack competitor data), Slow Path DAG modification, and evidence merging.

```
Step 1 — SUPERVISOR (PLAN — Slow Path)
  DAG: T1(researcher, "Infosys cloud Q3 revenue", group=1)
       ║ T2(researcher, "AWS Azure GCP cloud Q3 revenue", group=1)  ← PARALLEL
       → T3(skeptic) → T4(synthesizer)
  Decision: "continue", next_tasks = [T1, T2]

Step 2 — EXECUTOR → T1 ║ T2 (Parallel via Send())
  Returns [Send("executor", {next_tasks: [T1]}), Send("executor", {next_tasks: [T2]})]
  
  T1 (Worker 1): Researcher, HyDE → retrieval → 4 chunks relevant ✅
    Output: {chunks_after_grading: 4, pass_rate: 0.80, self_rag: "grounded"}
  
  T2 (Worker 2): Researcher, HyDE → retrieval → CRAG retries 3/3 → 1 chunk relevant ❌
    Output: {chunks_after_grading: 1, pass_rate: 0.10, self_rag: "partial"}
  
  evidence_board: [T1 chunks + T2 chunks] via evidence_reducer (deduped)

Step 3 — SUPERVISOR checks T1 (FAST PATH)
  T1.criteria: "≥2 chunks, pass_rate≥0.30, self_rag=grounded"
  chunks=4≥2 ✅  pass_rate=0.80≥0.30 ✅  grounded ✅
  T1: PASS

Step 4 — SUPERVISOR checks T2 (SLOW PATH — T2 FAILED criteria)
  T2.criteria: "≥2 chunks, pass_rate≥0.30, self_rag=grounded"
  chunks=1<2 ❌  pass_rate=0.10<0.30 ❌  partial≠grounded ❌
  
  LLM (gpt-4.1): "Internal documents don't contain competitor financials.
  Add web search task for missing data."
  
  MODIFIES DAG:
    T2.status = "failed"
    + T2b(web_search, "AWS Azure GCP cloud revenue Q3 2025",
          criteria: "≥1 relevant result, no timeout")
    T3.dependencies now: [T1, T2b] (not T2)
  
  Decision: "continue", next_tasks = [T2b]
  Cost: ~$0.015

Step 5 — EXECUTOR → T2b (Web Search → Tavily)
  Results: AWS $27.5B, Azure $24.1B, GCP $11.4B
  Content sanitized (no prompt injection)
  Output: {relevant_results: 3, timeout: False}

Step 6 — SUPERVISOR (FAST PATH)
  T2b.criteria: "≥1 relevant result, no timeout"
  relevant=3≥1 ✅  timeout=False ✅
  Next ready: T3 (T1=done, T2b=done)
  Decision: "continue", next_tasks = [T3]

Step 7 — EXECUTOR → T3 (Skeptic)
  Evidence: Infosys chunks (T1) + web data (T2b)
  NLI: "12% growth" vs "market share declined 3%" → CONTRADICTION (0.82)
  o3-mini: "Revenue grew 12% but market share fell — different metrics.
           Infosys grew slower than competitors who grew 15-20%."
  Output: {contradicted: 1, confidence: 0.72} — but explanation reconciles it

Step 8 — SUPERVISOR (SLOW PATH — contradiction detected)
  LLM: "Skeptic reconciled the contradiction. Revenue ≠ market share.
  Accept and proceed to synthesis."
  Decision: "continue", next_tasks = [T4]

Step 9 — EXECUTOR → T4 (Synthesizer)
  "Infosys cloud revenue grew 12% YoY [Annual Report, p.42]. 
   However, market share declined from 8.2% to 7.9% as competitors 
   grew faster: AWS $27.5B [Tavily], Azure $24.1B [Tavily], GCP $11.4B [Tavily]."

Step 10 — SUPERVISOR (FAST — all done): → VALIDATOR → PASS → END

TOTALS:
  Supervisor LLM calls: 3 (plan + T2 failure + contradiction)
  Supervisor Fast Path calls: 5 (free)
  Cost: ~$0.095
  Latency: ~22s
```

---

## Scenario 3: Contradictory Evidence

### Query: "What was AI's impact on Infosys's operating margins?"

**Why this scenario matters:** Tests the Skeptic's two-stage approach (NLI + LLM) and how conflicting evidence is reconciled, not suppressed.

```
Step 1 — SUPERVISOR (PLAN)
  DAG: T1(researcher, "AI impact on Infosys operating margins") 
       → T2(skeptic) → T3(synthesizer)

Step 2 — EXECUTOR → T1 (Researcher)
  Retrieves from different documents:
    chunk_12 (Annual Report 2026, p.78):
      "AI-driven automation improved operational efficiency by 2.3%"
    chunk_45 (Q3 Quarterly Report, p.15):
      "AI infrastructure investments compressed EBITDA margins by 1.8 
       percentage points during the transition period"
    chunk_67 (CEO Letter, p.3):
      "We expect AI to be margin-accretive within 18 months"
  
  Output: {chunks_after_grading: 3, pass_rate: 0.60, self_rag: "grounded"}

Step 3 — SUPERVISOR (FAST PATH — T1 criteria pass ✅)
  Decision: "continue", next_tasks = [T2]

Step 4 — EXECUTOR → T2 (Skeptic)
  Stage 1 — NLI (BART-MNLI, free, <100ms):
    "improved efficiency by 2.3%" paired with
    "compressed margins by 1.8pp"
    → CONTRADICTION (score 0.89)
    
    "margin-accretive within 18 months" paired with
    "compressed margins by 1.8pp"
    → NEUTRAL (score 0.71, future vs. current)
  
  Stage 2 — o3-mini judge:
    Analysis: "These are NOT contradictory. They describe DIFFERENT ASPECTS:
    - chunk_12: Revenue-side efficiency gains (2.3% improvement)
    - chunk_45: Cost-side investment impact (1.8pp compression)
    - chunk_67: Forward-looking expectation
    
    Net current impact: +2.3% (efficiency) - 1.8pp (investment) = +0.5% net
    
    RECONCILIATION: AI has dual margin impact — immediate gains partially 
    offset by infrastructure costs, expected to be fully accretive in 18 months."
    
    Verdict: RECONCILABLE
    Confidence: 0.78
  
  Output: SkepticOutput(
    claims_contradicted=1,  # Real NLI contradiction
    contradiction_details=["Revenue-side vs cost-side: reconciled"],
    logical_gaps=[],
    overall_confidence=0.78
  )

Step 5 — SUPERVISOR (SLOW PATH — contradiction count > 0)
  LLM: "Skeptic found 1 contradiction but reconciled it. The explanation 
  is logically sound (revenue-side vs cost-side). Accept reconciliation 
  and include both perspectives in synthesis."
  Decision: "continue", next_tasks = [T3]

Step 6 — EXECUTOR → T3 (Synthesizer, gpt-4.1)
  With critique notes from Skeptic:
  "AI had a dual impact on Infosys's operating margins in Q3 FY26:
  
  **Positive:** AI-driven automation improved operational efficiency by 2.3% 
  [Annual Report 2026, p.78].
  
  **Negative:** AI infrastructure investments compressed EBITDA margins by 1.8 
  percentage points [Q3 Quarterly Report, p.15].
  
  **Net effect:** Approximately +0.5% net margin improvement.
  
  **Outlook:** Management expects AI to become fully margin-accretive within 
  18 months [CEO Letter, p.3]."

Step 7 — VALIDATOR
  faithfulness: 0.93 ✅ (each claim traced to source)
  citation_accuracy: 0.97 ✅
  → PASS → END

KEY INSIGHT: The system presents BOTH sides with citations, not a false consensus.
  Cost: ~$0.065
```

---

## Scenario 4: Ambiguous Query

### Query: "How is the technology division performing?"

**Why this scenario matters:** Tests the Ambiguity Detector (pre-Supervisor) and mid-execution HITL `interrupt()`.

```
Step 0 — AMBIGUITY DETECTOR (Pre-Supervisor, gpt-4.1-mini)
  Analysis: "technology division" could mean:
  - Cloud Services
  - AI & ML Products
  - Enterprise Software
  - All of the above
  
  ambiguity_score: 0.84 > 0.70 threshold → AMBIGUOUS
  
  interrupt({
    type: "ambiguous_query",
    options: [
      "Cloud Services only",
      "AI & ML Products only",
      "Enterprise Software only",
      "All technology sub-divisions"
    ],
    suggestion: "Did you mean a specific sub-division?"
  })
  → State saved to PostgreSQL → API returns 202

... User responds: "Cloud Services only" ...

  graph.invoke(Command(resume={"selection": "Cloud Services only"}), config)
  → Graph resumes
  → Query refined to: "How is Infosys's Cloud Services division performing in Q3 FY26?"

Step 1 — SUPERVISOR (PLAN)
  DAG: T1(researcher, "Cloud Services Q3 FY26 performance revenue growth margins")
       → T2(skeptic) → T3(synthesizer)

Step 2 — EXECUTOR → T1 (Researcher)
  Retrieves: revenue, YoY growth, margin data for Cloud Services
  Output: {chunks_after_grading: 5, pass_rate: 0.50, self_rag: "grounded"}

Step 3 — SUPERVISOR (FAST ✅) → T2

Step 4 — EXECUTOR → T2 (Skeptic)
  Clean findings. Confidence 0.88. No contradictions.

Step 5 — SUPERVISOR (FAST ✅) → T3

Step 6 — EXECUTOR → T3 (Synthesizer)
  Well-cited analysis of Cloud Services performance.

Step 7 — VALIDATOR → PASS → END

TOTAL: 1 HITL pause + normal pipeline.
  Cost: ~$0.045 (no wasted research on wrong division)
```

---

## Scenario 5: Evidence Insufficient → HITL Mid-Execution

### Query: "Compare Infosys's R&D spending to industry benchmarks across 5 dimensions"

**Why this scenario matters:** Tests mid-execution HITL when research coverage is low.

```
Step 1 — SUPERVISOR (PLAN)
  DAG: T1(researcher, "Infosys R&D spend")
       ║ T2(researcher, "R&D as % of revenue")
       ║ T3(researcher, "R&D patent output")
       ║ T4(researcher, "R&D headcount trends")
       ║ T5(researcher, "R&D efficiency metrics")
       → T6(skeptic) → T7(synthesizer)

Step 2 — EXECUTOR → T1 ║ T2 ║ T3 ║ T4 ║ T5 (Parallel, 5 Send())
  T1: R&D total spend found → pass_rate 0.60 ✅
  T2: R&D/revenue ratio found → pass_rate 0.40 ✅
  T3: Patent data → CRAG 3/3 retries → pass_rate 0.05 ❌
  T4: Headcount → pass_rate 0.0, 0 relevant chunks ❌
  T5: Efficiency metrics → CRAG 3/3 retries → pass_rate 0.0 ❌

Step 3 — SUPERVISOR checks all 5 results:
  T1: FAST ✅ | T2: FAST ✅
  T3: SLOW → LLM: "No patent data in internal docs. Add web search."
  T4: SLOW → LLM: "No headcount data. Add web search."
  T5: SLOW → LLM: "No efficiency benchmarks. Try web search."
  
  Modified DAG:
    + T3b(web_search, "Infosys patent filings 2025-2026")
    + T4b(web_search, "Infosys R&D headcount LinkedIn")
    + T5b(web_search, "tech industry R&D efficiency benchmarks")

Step 4 — EXECUTOR → T3b ║ T4b ║ T5b (Parallel web searches)
  T3b: 2 patent-related articles found ✅
  T4b: No reliable headcount data ❌
  T5b: 1 industry benchmark report ✅

Step 5 — SUPERVISOR checks web results:
  T3b: FAST ✅ | T5b: FAST ✅
  T4b: SLOW → LLM evaluates coverage:
    "We have 4 of 5 dimensions (spend, ratio, patents, efficiency). 
     Headcount data NOT available from any source.
     Coverage: 4/5 = 80%. Below 100% but above 50%.
     But user explicitly asked for FIVE dimensions — need to ask."
  
  interrupt({
    type: "evidence_insufficient",
    summary: "Found evidence for 4 of 5 requested dimensions. 
              Headcount data unavailable from internal docs or web.",
    missing: ["R&D headcount trends"],
    coverage: 0.80,
    options: [
      "accept_partial: Proceed with 4/5 dimensions, note gap",
      "provide_data: Upload headcount data manually",
      "modify_scope: Reduce to 4 dimensions",
      "cancel"
    ],
  })
  → State saved → API returns 202

... User: "accept_partial" ...

  graph.invoke(Command(resume={"action": "accept_partial"}), config)
  → Supervisor: "User accepts partial coverage. Proceeding with 4 dimensions.
     Adding disclaimer about missing headcount data."
  → Dispatches T6(skeptic), then T7(synthesizer)

Step 6-8 — Normal: Skeptic ✅ → Synthesizer (with disclaimer) → Validator → PASS

TOTALS:
  Supervisor LLM calls: 5 (plan + 3 failures + coverage assessment)
  HITL pauses: 1
  Cost: ~$0.12
  Latency: ~35s + user think time
```

---

## Scenario 6: Infinite Loop Prevention

### Query: "Find evidence that Infosys's market share is declining"

**Why this scenario matters:** Tests THREE layers of loop prevention when evidence simply doesn't exist.

```
Step 1 — SUPERVISOR (PLAN)
  DAG: T1(researcher, "Infosys market share decline evidence")
       → T2(skeptic) → T3(synthesizer)

Step 2 — EXECUTOR → T1 (Researcher)
  HyDE rewrite → "Infosys's market share dropped from..."
  CRAG grading: all 5 chunks contradict premise → RETRY 1
  Rewrite: "Has Infosys lost competitive position?" → RETRY 2
  Rewrite: "Infosys market share trajectory" → RETRY 3
  All 3 retries: 0 relevant chunks supporting decline
  
  Output: {chunks_after_grading: 0, pass_rate: 0.0, self_rag: "not_grounded"}
  
  ⚡ LAYER 1: CRAG max retries (3/3) exhausted inside Eng 6 pipeline

Step 3 — SUPERVISOR (SLOW PATH — T1 criteria totally failed)
  LLM: "Internal documents contain no evidence of market share decline.
  Try web search for external analysis."
  Modified DAG: + T1b(web_search, "Infosys market share decline analysis 2025-2026")

Step 4 — EXECUTOR → T1b (Web Search)
  Tavily results: 0 relevant results (Infosys market share is actually GROWING)
  Output: {relevant_results: 0, timeout: False}

Step 5 — SUPERVISOR (FAST PATH)
  ⚡ LAYER 2: Cosine similarity check
  cosine(T1.query_embedding, T1b.query_embedding) = 0.91 > 0.90 threshold
  → Repetitive search detected
  
  Decision: "force_synthesize"
  Reason: "repetitive_loop_detected"

Step 6 — EXECUTOR → T_synth (Synthesizer, force mode)
  "Based on available evidence, there is NO data supporting a decline 
   in Infosys's market share. In contrast, available data indicates:
   - Q3 cloud revenue grew 12% YoY [Annual Report, p.42]
   - Cloud division grew 18% [Quarterly Report, p.15]
   
   If you have specific sources suggesting a market share decline, 
   please provide them for analysis."

Step 7 — VALIDATOR → PASS → END

LAYERS THAT PREVENTED INFINITE LOOP:
  1. CRAG: 3 retries inside Eng 6 pipeline (built-in)
  2. Supervisor: cosine > 0.9 → force stop (Fast Path, free)
  3. Hard cap: MAX_SUPERVISOR_TURNS = 15 (if layers 1+2 somehow failed)

Cost: ~$0.042
KEY INSIGHT: System gives honest "no evidence found" answer rather than hallucinating.
```

---

## Scenario 7: Circuit Breaker & Model Fallback

### Query: "Summarize Infosys's sustainability initiatives"

**Why this scenario matters:** Tests API failure resilience with 3-state circuit breaker and model fallback chains.

```
Step 1 — SUPERVISOR (PLAN)
  DAG: T1(researcher, "Infosys sustainability initiatives")
       → T2(skeptic) → T3(synthesizer)

Step 2 — EXECUTOR → T1 (Researcher, gpt-4.1-mini)
  HyDE rewrite → API call → OpenAI 429 Too Many Requests
  
  Circuit breaker state: CLOSED → records failure 1/4
  Retry with exponential backoff (1s, 2s, 4s)
  Attempt 2: 429 again → failure 2/4
  Attempt 3: 429 again → failure 3/4
  Attempt 4: 429 again → failure 4/4
  
  ⚡ Circuit breaker: CLOSED → OPEN (4 failures reached threshold)
  
  MODEL FALLBACK:
  Researcher fallback chain: gpt-4.1-mini → gpt-4.1 → return partial
  
  Switches to gpt-4.1 → SUCCESS
  HyDE rewrite works, pipeline continues with gpt-4.1 as researcher model
  
  Output: {chunks_after_grading: 3, pass_rate: 0.60, self_rag: "grounded"}
  (Slightly higher cost due to gpt-4.1, but transparent to Supervisor)

Step 3 — SUPERVISOR (FAST ✅ — doesn't know about the fallback)

Step 4 — EXECUTOR → T2 (Skeptic, o3-mini)
  o3-mini API works fine. No issues.
  Output: {confidence: 0.85}

Step 5 — SUPERVISOR (FAST ✅)

Step 6 — EXECUTOR → T3 (Synthesizer, gpt-4.1)
  Works fine. Clean answer about sustainability.

... 60 seconds later, circuit breaker: OPEN → HALF-OPEN ...
  Probe: test call to gpt-4.1-mini → SUCCESS
  Circuit breaker: HALF-OPEN → CLOSED (back to normal for next query)

TOTALS:
  Cost: ~$0.065 (higher due to gpt-4.1 fallback for researcher)
  Latency: ~16s (4 retries added ~7s)
  User impact: NONE — transparent model switch, answer quality maintained
```

---

## Scenario 8: Validator Loop-Back

### Query: "What is Infosys's strategy for emerging markets?"

**Why this scenario matters:** Tests the Validator → Supervisor → re-research → re-synthesize loop.

```
Steps 1-6 — Normal flow: Plan → Research → Skeptic → Synthesizer

Step 7 — VALIDATOR
  faithfulness: 0.71 ❌ (threshold: 0.85)
  citation_accuracy: 0.88 ❌ (threshold: 0.90)
  answer_relevancy: 0.82 ✅
  dag_completeness: 0.95 ✅
  
  → route_validator: "revise" → back to SUPERVISOR

Step 8 — SUPERVISOR (SLOW PATH — receives quality_scores)
  LLM: "Faithfulness is low. The claim 'Infosys plans to enter 12 new markets' 
  appears unsupported — only 8 markets mentioned in evidence. And 1 citation 
  references a chunk not in the evidence board.
  
  Adding T5(researcher, 'Infosys emerging markets expansion plan details')
  to get more specific data, then re-synthesize."
  
  Modified DAG: + T5(researcher) → T6(synthesizer, "re-synthesize with verified data")

Step 9 — EXECUTOR → T5 (Researcher): finds 4 additional chunks about emerging markets

Step 10 — SUPERVISOR (FAST ✅): → T6

Step 11 — EXECUTOR → T6 (Synthesizer): re-generates with corrected data
  "Infosys plans to enter 8 new emerging markets in FY27 [Strategy Doc, p.23]..."

Step 12 — SUPERVISOR (FAST — complete): → VALIDATOR

Step 13 — VALIDATOR (Round 2)
  faithfulness: 0.93 ✅
  citation_accuracy: 0.96 ✅
  → PASS → END

TOTALS:
  Validator rounds: 2 (first failed, second passed)
  Cost: ~$0.085
  KEY: Self-correcting system — hallucinated "12 markets" → verified "8 markets"
```

---

## Scenario 9: Full DAG With User Editing

### Query: "Provide a comprehensive analysis of Infosys's competitive position"

**Why this scenario matters:** Tests the complete DAG lifecycle including user editing via `interrupt()` and `Command(resume=...)`.

```
Step 1 — SUPERVISOR (PLAN)
  Initial DAG:
    T1(researcher, "Infosys revenue & growth trends", group=1)
    ║ T2(researcher, "Infosys market share by segment", group=1)
    ║ T3(web_search, "competitor analysis tech sector 2025", group=1)
    → T4(skeptic, "cross-check all evidence", group=2)
    → T5(synthesizer, "competitive position analysis", group=3)
  
  ⚡ interrupt() — Pause for user approval of DAG
  interrupt({
    type: "dag_approval",
    planned_dag: [T1, T2, T3, T4, T5],
    estimated_cost: "$0.08",
    estimated_time: "25s",
    message: "Here's my research plan. You can modify, add, or remove tasks."
  })
  → API returns 202

... User reviews and modifies ...

  User: POST /resume {
    thread_id: "xyz",
    action: "modify_dag",
    modifications: {
      add: [{"type": "researcher", "query": "Infosys patent portfolio strength"}],
      remove: [],
      modify: [{"task_id": "T3", "query": "AWS Azure Google competitive analysis 2025-2026"}]
    }
  }

  graph.invoke(Command(resume=user_modifications), config)

Step 2 — SUPERVISOR receives resume value
  interrupt() returns the user's modifications
  Applied changes:
    T3.query updated to "AWS Azure Google competitive analysis 2025-2026"
    + T2b(researcher, "Infosys patent portfolio strength", group=1)
  
  Decision: "continue", next_tasks = [T1, T2, T2b, T3] (all group=1)

Step 3 — EXECUTOR → T1 ║ T2 ║ T2b ║ T3 (4 parallel tasks via Send())
  T1: 5 chunks ✅ | T2: 3 chunks ✅ | T2b: 4 chunks ✅ | T3: 3 web results ✅

Steps 4-7 — Normal: Fast Path routing → Skeptic → Synthesizer → Validator → PASS

TOTALS:
  DAG editing pauses: 1
  Parallel workers: 4
  Cost: ~$0.11
  User influence: Added patent analysis, refined competitor query
```

---

## Scenario 10: Budget Exhaustion → Graceful Degradation

### Query: Very complex multi-dimensional analysis (5 research + 3 web + 2 skeptic + 2 synthesizer)

**Why this scenario matters:** Tests budget enforcement and graceful partial results.

```
Steps 1-12 — Normal execution consuming tokens...
  token_budget.total_tokens_used: 82,000 of 100,000

Step 13 — SUPERVISOR (FAST PATH)
  Check 1: budget remaining = 18,000 tokens
  This is enough for maybe 1 more researcher, but not a full skeptic + synthesizer
  
  Continue cautiously. Next task: T8(researcher)

Step 14 — EXECUTOR → T8 (Researcher) 
  Uses 12,000 tokens
  Budget remaining: 6,000

Step 15 — SUPERVISOR (FAST PATH)
  Check 1: budget remaining = 6,000 ← BELOW synthesizer minimum (8,000 estimated)
  
  ⚡ Decision: "force_synthesize"
  Reason: "budget_insufficient_for_full_pipeline"
  
  Skips remaining T9(skeptic), goes directly to synthesizer with budget caveat

Step 16 — EXECUTOR → T_synth (force-synthesize, gpt-4.1, budget-aware prompt)
  "Based on analysis of 8 research dimensions:
   [... answer with available evidence ...]
   
   NOTE: This analysis covers 8 of 10 planned dimensions. The following 
   were not fully analyzed due to resource constraints:
   - Employee sentiment trends
   - Regulatory risk assessment
   
   Token budget: 94,000/100,000 used."

Step 17 — VALIDATOR → PASS (with lowered completeness threshold for forced synthesis)

TOTALS:
  Budget used: 94,000/100,000 tokens
  Dimensions covered: 8/10
  Answer quality: GOOD but incomplete
  Key: System didn't crash — gave best answer within budget
```

---

## Summary: What Each Scenario Tests

| # | Scenario | Key Feature Tested |
|---|---|---|
| S1 | Simple factual | Happy path, Fast Path dominance (4 free checks) |
| S2 | Multi-step failure | Send() parallel, Slow Path DAG modification, web fallback |
| S3 | Contradictory evidence | NLI + LLM reconciliation, nuanced answer |
| S4 | Ambiguous query | Pre-supervisor interrupt(), user clarification |
| S5 | Evidence insufficient | Mid-execution HITL, partial coverage decision |
| S6 | Infinite loop | 3-layer prevention (CRAG + cosine + hard cap) |
| S7 | Circuit breaker | Model fallback transparency |
| S8 | Validator loop-back | Quality-driven re-research and re-synthesis |
| S9 | User DAG editing | Full DAG lifecycle with user modification |
| S10 | Budget exhaustion | Graceful degradation with partial results |

---

## Architecture Deep-Dive: How Our Design Solves Each Concern

### Q1: How does the Supervisor decompose the user query into a task DAG?

**Mechanism:** On the **first turn** (`iteration_count == 0`), the Supervisor calls gpt-4.1 with `with_structured_output(TaskPlan)`. The prompt includes few-shot examples of good decompositions.

**Example:**

```
User: "Compare Infosys's cloud revenue to competitors and analyze growth trends"

Supervisor prompt (simplified):
  "You are a research planner. Decompose this query into tasks.
   Each task has: type, query, dependencies, parallel_group, acceptance_criteria.
   
   FEW-SHOT EXAMPLE:
   Query: 'What causes climate change?'
   Plan: T1(researcher, 'greenhouse gas emissions data', group=1)
         ║ T2(researcher, 'historical temperature trends', group=1)
         → T3(skeptic, 'cross-check findings', group=2)
         → T4(synthesizer, 'explain causes of climate change', group=3)"

gpt-4.1 output (structured, Pydantic-enforced):
  TaskPlan(
    tasks=[
      T1(researcher, "Infosys cloud Q3 revenue", group=1, 
         criteria="≥2 chunks, pass_rate≥0.30, grounded"),
      T2(web_search, "AWS Azure GCP Q3 revenue", group=1,
         criteria="≥1 relevant result"),
      T3(researcher, "Infosys cloud growth trend 5 years", group=1,
         criteria="≥2 chunks, pass_rate≥0.30, grounded"),
      T4(skeptic, "cross-check all evidence", group=2,
         criteria="0 contradicted, confidence≥0.65"),
      T5(synthesizer, "comparison + trend analysis", group=3,
         criteria="all claims cited"),
    ])
```

**Key:** The LLM writes `acceptance_criteria` as a string for each task. The Fast Path parses and checks these against the agent's structured output fields. This is how the Supervisor "tells each agent what success looks like" at plan time.

---

### Q2: How does the Supervisor monitor execution state?

**Mechanism:** The Supervisor runs **after every single task** (because `executor → supervisor` is a hard edge). It reads `last_task_result` from state.

**Example — Step by step monitoring:**

```
Iteration 1: Supervisor PLANS → dispatches T1
Iteration 2: T1 done → Supervisor reads T1.output → Fast Path → PASS → dispatch T2
Iteration 3: T2 done → Supervisor reads T2.output → Fast Path → FAIL (pass_rate 0.10)
             → Slow Path → LLM decides: "Modify DAG, add T2b(web_search)"
Iteration 4: T2b done → Supervisor reads T2b.output → Fast Path → PASS → dispatch T3
...
```

**What the Supervisor sees (context control):**

```python
# Supervisor gets ONLY:
supervisor_view = {
    "original_query": state["original_query"],
    "task_dag": state["task_dag"],           # Status of all tasks
    "last_task_result": {
        "task_id": "T2",
        "summary": result.summary[:500],     # 200 tokens, NOT full evidence
        "status": "failed",
        "criteria_check": {"pass_rate": 0.10, "threshold": 0.30, "verdict": "FAIL"},
    },
    "iteration_count": 3,
    "token_budget": {"used": 12000, "remaining": 88000},
}
# Supervisor NEVER sees full evidence chunks → prevents context bloat
```

---

### Q3: How does the Supervisor decide when to retry, escalate, or stop?

**Mechanism:** Two-tier decision (Fast Path → Slow Path).

```
Fast Path (free, <10ms, deterministic):
  ┌─ Budget exhausted? → force_synthesize
  ├─ Iteration limit (15)? → force_synthesize
  ├─ Repetitive search (cosine > 0.9)? → force_synthesize
  └─ Agent criteria PASS? → dispatch next ready task
     If no more ready tasks → ready_for_validation

Slow Path (gpt-4.1, ~$0.015, when Fast Path criteria FAIL):
  LLM sees: failed_task + task_dag + budget + iteration_count
  LLM returns structured decision:
    retry: {new_query: "...", max_retries: 1}
    modify_dag: {add: [...], remove: [...], update_deps: [...]}
    escalate: {reason: "...", hitl_options: [...]}
    force_synthesize: {reason: "enough evidence for partial answer"}
    stop: {reason: "unrecoverable failure"}
```

**Example — Retry vs Escalate decision:**

```
CASE A: Researcher returns pass_rate=0.15 (close to 0.30)
  Slow Path LLM: "Near-miss. Rewrite query and retry."
  → retry with modified query, crag_retries_remaining > 0

CASE B: Skeptic returns confidence=0.38, contradictions=3
  Slow Path LLM: "Contradictions are severe and I cannot resolve them.
  Escalating to human."
  → interrupt({type: "contradictions", details: [...], options: [...]})
```

---

### Q4: How does the Skeptic (critic) actively search for hallucinations and logical gaps?

**Mechanism:** Two-stage approach — cheap NLI first, expensive LLM second.

**Example:**

```
Evidence board after research:
  chunk_12: "Revenue grew 12% to ₹41,764 crore"
  chunk_45: "Operating margins compressed by 1.8pp"
  chunk_67: "AI investments expected to pay off in 18 months"

STAGE 1 — NLI (BART-MNLI, local, free, <100ms per claim):
  For each claim in evidence, check against source text:
  
  Claim: "Revenue grew 12%"
  Source: chunk_12 text
  NLI: ENTAILMENT (0.94) ✅ — claim is supported
  
  Claim: "Margins improved due to AI"
  Source: chunk_45 text ("margins COMPRESSED")
  NLI: CONTRADICTION (0.89) ❌ — claim says improved, source says compressed
  
  → Flag: contradiction detected between chunk_12 implication and chunk_45

STAGE 2 — LLM Judge (o3-mini, ~$0.008):
  Receives: all evidence + NLI flags
  Prompt enforces: "You MUST find at least 3 issues. Be adversarial."
  
  Output:
    contradictions: ["Margins: chunk_12 implies positive, chunk_45 shows negative"]
    logical_gaps: ["No data on absolute margin values, only relative changes"]
    weak_evidence: ["chunk_67 is forward-looking, not factual"]
    confidence: 0.72
```

**Why o3-mini instead of gpt-4.1?** Anti-sycophancy — using a different model family from the Synthesizer ensures the critic won't "agree" with the answer generator.

---

### Q5: How does the Skeptic challenge assumptions and weak evidence?

**Mechanism:** The Skeptic prompt explicitly requires adversarial analysis:

```python
SKEPTIC_PROMPT = """You are a research auditor. Your job is to CHALLENGE evidence.

RULES:
1. You MUST identify at least 3 potential issues (even minor ones)
2. Flag any claim that relies on a single source
3. Flag any forward-looking statement used as fact
4. Flag any claim where the source is older than 2 years
5. Flag any logical leap between evidence and conclusion

For each issue, classify:
  - CRITICAL: contradicts source evidence
  - WARNING: weak evidence, single-source, or outdated
  - INFO: minor concern, acceptable with disclaimer
"""
```

**Example — Single-source weakness:**

```
Evidence: Only chunk_12 mentions "12% growth"
Skeptic: WARNING — "12% growth claim relies on single source (Annual Report p.42).
         No corroboration from quarterly reports or press releases.
         Recommendation: search for confirming evidence."
```

---

### Q6: How does the Synthesizer produce the final response?

**Mechanism:** Receives evidence board + Skeptic critique + task DAG. Uses gpt-4.1 with Pydantic structured output.

**Example:**

```python
# Input to Synthesizer:
evidence = [chunk_12, chunk_24, chunk_38, chunk_45]  # U-shape ordered
critique = SkepticOutput(contradictions=[...], weak_flags=[...])
task_dag = [T1(done), T2(done), T3(done)]

# U-shape ordering: most relevant at START and END (where LLMs attend best)
ordered = [chunk_12(best), chunk_45(worst), chunk_24(mid), chunk_38(2nd best)]

# gpt-4.1 generates with Pydantic enforcement:
SynthesizerOutput(
    answer="Infosys's Q3 revenue was ₹41,764 crore (+12% YoY) [1]. 
    Operating margins compressed 1.8pp due to AI investments [2], 
    but efficiency gains of 2.3% partially offset this [3].",
    citations=[
        Citation(chunk_id="c12", claim_text="₹41,764 crore, +12%"),
        Citation(chunk_id="c45", claim_text="compressed 1.8pp"),
        Citation(chunk_id="c12", claim_text="efficiency gains 2.3%"),
    ],
    claims_count=3,
    citations_count=3,
    all_citations_in_evidence_board=True,
)
```

---

### Q7: How does the Synthesizer include citations and justification?

**Mechanism:** Pydantic schema **makes uncited claims structurally impossible.**

```python
class SynthesizerOutput(BaseModel):
    answer: str
    citations: list[Citation] = Field(min_length=1)  # ← CANNOT be empty
    claims_count: int
    citations_count: int
    all_citations_in_evidence_board: bool

class Citation(BaseModel):
    chunk_id: str           # Must match an EvidenceChunk in evidence_board
    claim_text: str         # The exact claim this citation supports
    entailment_score: float # Filled post-hoc by NLI verifier
```

**Validator double-checks:** Even if Synthesizer claims `all_citations_in_evidence_board=True`, the Validator independently verifies each citation's `chunk_id` exists in `evidence_board` and NLI-scores the entailment. If `citation_accuracy < 0.90` → revise loop.

---

### Q8: How does Few-Shot Prompting guide task decomposition?

**Mechanism:** The Supervisor's planning prompt includes 3-4 few-shot examples of good DAG decompositions covering different query types.

```python
SUPERVISOR_PLAN_PROMPT = """
Decompose the user's query into a research task plan.

EXAMPLE 1 (Simple factual):
  Query: "What was Q3 revenue?"
  Plan: T1(researcher, "Q3 revenue") → T2(skeptic) → T3(synthesizer)

EXAMPLE 2 (Comparative, needs web):
  Query: "Compare our revenue to competitors"
  Plan: T1(researcher, "our revenue", group=1) 
        ║ T2(web_search, "competitor revenue", group=1)
        → T3(skeptic) → T4(synthesizer)

EXAMPLE 3 (Multi-dimensional):
  Query: "Analyze impact of AI on operations, margins, and headcount"
  Plan: T1(researcher, "AI operations impact", group=1)
        ║ T2(researcher, "AI margin impact", group=1)
        ║ T3(researcher, "AI headcount impact", group=1)
        → T4(skeptic) → T5(synthesizer)

EXAMPLE 4 (Thematic, broad):
  Query: "What are the key risks facing the company?"
  Plan: T1(researcher, "financial risks") ║ T2(researcher, "operational risks")
        ║ T3(researcher, "market risks") ║ T4(web_search, "industry risk trends")
        → T5(skeptic) → T6(synthesizer)

Now decompose: "{user_query}"
Each task MUST include acceptance_criteria.
"""
```

---

### Q9: How does the system handle invalid or hallucinated tool calls?

**Mechanism:** Three layers prevent invalid calls.

```
Layer 1 — Pydantic Structured Output:
  Supervisor returns TaskPlan via with_structured_output(TaskPlan).
  Invalid task types, missing fields → Pydantic ValidationError → caught → retry.
  
  Example: LLM tries type="analyzer" (not in Literal["researcher","web_search","skeptic","synthesizer"])
  → Pydantic rejects → retry with corrected prompt

Layer 2 — Executor Dispatch Guard:
  if task.type not in VALID_TYPES:
      return AgentError(task_id=task.task_id, error="unknown_agent_type", 
                        suggestion="Valid types: researcher, web_search, skeptic, synthesizer")

Layer 3 — Supervisor Slow Path:
  If an agent returns an error, Supervisor gets it as last_task_result.
  LLM reads the error and decides: retry with corrected task, skip, or escalate.
```

---

### Q10: How does State Management prevent uncontrolled context growth?

**Mechanism:** Per-agent filtered views + Supervisor sees only summaries.

```
Problem: 10 research tasks × 5 chunks each = 50 chunks × 2000 chars = 100,000 chars in state

Solution:
  1. Supervisor NEVER sees full evidence_board
     → Only sees last_task_result.summary (200 tokens / ~500 chars)
     
  2. Each Researcher sees ONLY its own TaskNode.query
     → NOT other researchers' evidence
     
  3. Skeptic sees ALL evidence (by design — needs cross-document analysis)
     → But runs ONCE, not repeatedly
     
  4. Synthesizer sees evidence_board but with U-shape ordering
     → Best chunks at start+end, weakest in middle
     
  5. evidence_reducer deduplicates by (doc_id, chunk_id)
     → Same chunk retrieved by 2 researchers → stored only once (best score)

Result: State grows linearly with unique evidence, NOT with task count.
```

---

### Q11: How does Short-Term Memory act as a shared "whiteboard"?

**Mechanism:** The `evidence_board` IS the whiteboard. All agents read from and write to it.

```
T1(researcher) writes: [chunk_12, chunk_24] → evidence_board
T2(researcher) writes: [chunk_45, chunk_67] → evidence_board (via reducer, deduped)
T3(skeptic) READS: all 4 chunks from evidence_board
T3(skeptic) writes: critique_notes (separate field)
T4(synthesizer) READS: evidence_board + critique_notes
T4(synthesizer) writes: synthesis_output

Whiteboard at end: {
  evidence_board: [chunk_12, chunk_24, chunk_45, chunk_67],  ← shared
  critique_notes: [SkepticOutput(...)],                       ← shared
  synthesis_output: SynthesizerOutput(...),                    ← shared
}
```

**Agents collaborate without direct communication** — they share via state.

---

### Q12: How does Routing Logic handle irrelevant or malformed queries?

**Mechanism:** Ambiguity Detector (pre-Supervisor) classifies the query.

```
Query: "What's the weather today?"
→ Ambiguity Detector: OUT_OF_SCOPE (0.95)
→ Direct response: "I can only answer questions about Infosys's business 
   documents. For weather, please use a weather service."
→ NEVER reaches Supervisor → no DAG created → no cost

Query: "How is things going?"
→ Ambiguity Detector: AMBIGUOUS (0.84)
→ interrupt({options: ["Cloud Services?", "Financial Services?", "Overall?"]})

Query: "What was Q3 revenue?"
→ Ambiguity Detector: CLEAR (0.12)
→ Proceeds to Supervisor normally
```

---

### Q13: How does Rate Limiting prevent API exhaustion?

**Mechanism:** Checked in Fast Path (free, every turn) + per-agent limits.

```python
TOOL_LIMITS = {
    "researcher":   {"max_parallel": 3, "max_total": 8,  "timeout_s": 30},
    "web_search":   {"max_parallel": 2, "max_total": 4,  "timeout_s": 15},
    "skeptic":      {"max_parallel": 1, "max_total": 3,  "timeout_s": 45},
    "synthesizer":  {"max_parallel": 1, "max_total": 3,  "timeout_s": 60},
}

# Example: Supervisor tries to dispatch 5 researcher tasks in parallel
# check_rate_limits() reduces to max_parallel=3:
next_tasks = [T1, T2, T3]  # Only 3 dispatched
remaining = [T4, T5]       # Queued for next iteration
```

**Budget enforcement (also in Fast Path):**
```
Every Supervisor turn checks:
  tokens_remaining > 0?      → if not, force_synthesize
  cost_remaining > $0?       → if not, force_synthesize
  wall_clock < 300s?          → if not, force_synthesize
  api_calls["researcher"] < 8? → if not, skip remaining researcher tasks
```

---

### Q14: How to swap GPT-4-class models for smaller SLMs?

**Mechanism:** `config/model_routing.py` centralizes all assignments.

```python
# config/model_routing.py
MODEL_ROUTING = {
    "supervisor_plan":    os.getenv("MODEL_SUPERVISOR", "gpt-4.1"),
    "supervisor_slow":    os.getenv("MODEL_SUPERVISOR", "gpt-4.1"),
    "researcher":         os.getenv("MODEL_RESEARCHER", "gpt-4.1-mini"),
    "skeptic_llm":        os.getenv("MODEL_SKEPTIC", "o3-mini"),
    "synthesizer":        os.getenv("MODEL_SYNTHESIZER", "gpt-4.1"),
    "ambiguity_detector": os.getenv("MODEL_AMBIGUITY", "gpt-4.1-mini"),
}

# To swap for smaller models — just change .env:
# MODEL_RESEARCHER=gpt-4.1-nano
# MODEL_AMBIGUITY=gpt-4.1-nano
# MODEL_SUPERVISOR=gpt-4.1-mini  (risky — supervisor quality drops)

# Or per-query at runtime via config:
result = graph.invoke(
    {"original_query": "...", "model_overrides": {"researcher": "gpt-4.1-nano"}},
    config
)
```

**Trade-offs (documented in config):**

| Role | Swap Safe? | Risk |
|---|---|---|
| Researcher | ✅ Yes — gpt-4.1-mini → gpt-4.1-nano | Lower grading quality, more CRAG retries |
| Ambiguity Detector | ✅ Yes | Slightly more false positives |
| Skeptic LLM | ⚠️ Careful | Cheaper model might miss contradictions |
| Supervisor | ❌ Not recommended | Bad DAG planning cascades to everything |
| Synthesizer | ❌ Not recommended | User-facing quality drops |

---

### Q15: What happens if the Researcher enters an infinite search loop?

**Answer:** Three layers prevent this (see Scenario 6 for full trace).

```
Layer 1 — CRAG (inside Eng 6): Max 3 rewrite-retrieve-grade retries.
  After 3 failures → returns what it has (even if 0 chunks).

Layer 2 — Supervisor Fast Path: cosine_similarity(T1.query, T1_retry.query) > 0.90
  → "Repetitive search detected" → force_synthesize.

Layer 3 — Hard cap: MAX_SUPERVISOR_TURNS = 15.
  Even if cosine check fails, absolute turn limit stops execution.

Example: "Find evidence of market share decline" (none exists)
  T1: CRAG 3/3 → 0 chunks → Supervisor adds T1b(web_search)
  T1b: 0 results → cosine(T1, T1b) = 0.91 > 0.90 → FORCE STOP
  → Synthesizer: "No evidence supports this claim. Available data shows growth."
```

---

### Q16: How do you prevent agentic drift from the original user intent?

**Mechanism:** Every component ties back to `original_query`.

```
1. original_query is IMMUTABLE — never modified in state
2. Each TaskNode.query is derived from original_query by Supervisor
3. Validator checks answer_relevancy(synthesis.answer, original_query) ≥ 0.80
4. Supervisor only accepts DAG modifications that contribute to stop_condition

Example drift prevention:
  Query: "Q3 revenue?"
  T1(researcher, "Q3 revenue") → finds mention of "new product launch"
  
  Supervisor (FAST): Does NOT auto-create T2(researcher, "new product launch")
    because it's outside stop_condition="Q3 revenue figure"
  
  Supervisor only adds tasks that serve the original_query's stop_condition.
```

---

### Q17: With 10,000 documents, how do you avoid losing critical context in RAG?

**Mechanism:** Parent-child chunking + metadata filtering + hybrid retrieval.

```
10,000 docs → Ingestion:
  Each doc split into parent chunks (2000 chars) and child chunks (500 chars)
  Child chunks embedded (text-embedding-3-small) → ChromaDB
  parent_id links child → parent

Query time:
  1. Metadata filter: year=2026, quarter=Q3 → reduces to ~200 docs
  2. Hybrid search on CHILD chunks (precise, 500 chars):
     Vector top 10 + BM25 top 10 → RRF fusion → 10 unique chunks
  3. Cross-encoder rerank: top 5 CHILD chunks
  4. For each top child → retrieve PARENT chunk (2000 chars) for full context
  
Result: Query searches 10,000 docs but only processes 5 × 2000 = 10,000 chars.
The metadata filter + hybrid search ensures the RIGHT 5 chunks are found.
```

---

### Q18: How do you handle 50 retrieved chunks without "lost-in-the-middle" issues?

**Mechanism:** We never pass 50 chunks to an LLM. Multiple layers reduce the count.

```
Retrieval: 10 chunks (hybrid search)
Rerank: 5 chunks (cross-encoder filter)
CRAG grading: 2-4 chunks survive (relevance grading)
Self-RAG: confirms grounding

Even in worst case (5 research tasks × 5 chunks = 25 chunks for Synthesizer):
  → U-shape ordering: best chunks at start + end of context
  → Puts strongest evidence where LLM attention is highest
  → Weakest chunks in middle (least attention needed for them)

def u_shape_order(chunks):
    sorted_chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    result = []
    for i, chunk in enumerate(sorted_chunks):
        if i % 2 == 0:
            result.insert(0, chunk)   # Best → start
        else:
            result.append(chunk)       # 2nd best → end
    return result
```

---

### Q19: How do you ensure repeated queries produce stable conclusions?

**Mechanism:** Deterministic components + LLM temperature control + caching.

```
Deterministic (same result every time):
  - BM25 search (pure scoring)
  - Cross-encoder rerank (deterministic model)
  - NLI entailment checks (BART-MNLI, deterministic)
  - Fast Path decisions (pure rules)
  - Evidence reducer (deterministic dedup)

LLM temperature set low:
  - Researcher: temp=0.1 (factual, minimal variation)
  - Supervisor plan: temp=0.2 (slight variation acceptable in task decomposition)
  - Synthesizer: temp=0.1 (factual, deterministic-ish)

Prompt caching:
  - System prompts cached across queries → consistent behavior
  - Few-shot examples are identical → stable decomposition patterns

Result: Same query → same retrieval → same reranking → similar LLM output.
Not 100% identical (LLMs are inherently stochastic) but stable within ±5%.
```

---

### Q20: Do they handle subtask failures? Fallback/retry strategy?

**Answer:** Yes. Every subtask failure routes to Supervisor Slow Path.

```
Retry strategy per agent:
  Researcher: CRAG max 3 retries (internal) → then Supervisor can retry with new query
  Web Search: Timeout → Supervisor retries with simplified query or skips
  Skeptic: API failure → model fallback (o3-mini → gpt-4.1 → skip with warning)
  Synthesizer: citation failure → Supervisor re-dispatches with stricter prompt

Example — Cascading failure handling:
  T1(researcher): pass_rate=0.05 → SLOW PATH → retry with modified query
  T1_retry(researcher): pass_rate=0.08 → SLOW PATH → "switch to web search"
  T1b(web_search): timeout → SLOW PATH → force_synthesize with partial evidence
  
  System never crashes. Worst case → partial answer with disclaimer.
```

---

### Q21: Do they acknowledge the expense and slowness of multi-agent loops?

**Answer:** Yes. This is why the Two-Tier design exists.

```
Without Two-Tier: Every supervisor turn = 1 LLM call = ~$0.015 × 10 turns = $0.15
With Two-Tier:    60-70% of turns are Fast Path ($0) → only 3-4 Slow Path calls

Cost comparison per typical query:
  Naive approach (LLM every turn): ~$0.15-0.20
  Our approach (Fast/Slow):         ~$0.04-0.06

Latency comparison:
  Naive: 10 turns × 3s = 30s just for supervisor decisions
  Ours:  4 Fast Path (0ms) + 2 Slow Path (6s) + agent time (15s) = ~21s

Budget enforcement: MAX cost $0.50/query, MAX 100K tokens, MAX 300s wall clock.
If approaching limits → force_synthesize with best available evidence.
```

---

### Q22: Can the system reason through contradictory evidence (Doc A vs Doc B)?

**Answer:** Yes. This is the Skeptic's primary purpose. See Scenario 3 for full trace.

```
Doc A (Annual Report): "AI improved efficiency by 2.3%"
Doc B (Q3 Quarterly): "AI compressed margins by 1.8pp"

Skeptic Stage 1 (NLI): CONTRADICTION detected (score 0.89)
Skeptic Stage 2 (o3-mini): "Not a real contradiction — different metrics.
  Doc A = revenue-side efficiency. Doc B = cost-side investment.
  Net impact: +2.3% - 1.8pp = +0.5% positive."

Synthesizer receives critique → presents BOTH perspectives:
  "AI had dual impact: efficiency gains of 2.3% [Doc A] offset by
   infrastructure costs of 1.8pp [Doc B]. Net: +0.5%."

The system does NOT suppress contradictions — it RECONCILES them with explanation.
If reconciliation is impossible → HITL escalation.
```

---

## Final Assessment: Is the Architecture Complete?

### What Is Solved ✅

| Concern | Status | How |
|---|---|---|
| Query decomposition | ✅ Solved | Supervisor + few-shot + Pydantic TaskPlan |
| Execution monitoring | ✅ Solved | Supervisor runs after every task (hard edge) |
| Retry/escalate/stop decisions | ✅ Solved | Two-tier Fast/Slow Path |
| Hallucination detection | ✅ Solved | NLI pre-filter + LLM judge + Self-RAG |
| Weak evidence challenges | ✅ Solved | Skeptic with adversarial prompt |
| Citations and justification | ✅ Solved | Pydantic min_length=1 + NLI verification |
| Few-shot prompting | ✅ Solved | 4 examples in supervisor planning prompt |
| Invalid tool calls | ✅ Solved | Pydantic validation + executor guard + retry |
| Context growth control | ✅ Solved | Filtered views + summary-only for Supervisor |
| Shared whiteboard | ✅ Solved | evidence_board + critique_notes in state |
| Irrelevant queries | ✅ Solved | Ambiguity Detector pre-gate |
| Rate limiting | ✅ Solved | Per-agent limits + budget enforcement |
| Model swapping | ✅ Solved | Centralized config + env vars |
| Infinite loops | ✅ Solved | 3-layer prevention (CRAG + cosine + hard cap) |
| Agentic drift | ✅ Solved | Immutable original_query + relevancy check |
| Lost-in-the-middle | ✅ Solved | U-shape ordering + chunk reduction pipeline |
| 10K doc scale | ✅ Solved | Metadata filter + hybrid search + parent-child |
| Subtask failures | ✅ Solved | Supervisor Slow Path + model fallback chains |
| Cost awareness | ✅ Solved | Two-tier design, 60-70% of routing is free |
| Contradictory evidence | ✅ Solved | NLI + LLM reconciliation, presents both sides |
| Stable results | ✅ Solved | Low temperature + deterministic components |
| HITL integration | ✅ Solved | interrupt() + Command(resume=...) |
| Fault tolerance | ✅ Solved | PostgresSaver + circuit breaker |

### Suggested Minor Enhancements (Not Architectural Changes)

These are **implementation-time optimizations**, NOT design changes:

1. **Prompt caching** — Cache system prompts across queries to reduce latency and cost. Can be added in `config/settings.py` without any architecture changes.

2. **Streaming intermediate results** — SSE events already defined in API layer. Implementation detail, not design.

3. **Observability dashboards** — Langfuse/LangSmith tracing. Plugin, not architecture.

### Verdict

> **The current 3-node architecture (Supervisor → Executor → Validator) with per-task acceptance_criteria and two-tier Fast/Slow Path addresses ALL concerns listed above. No design changes are needed. The architecture is ready for implementation starting with Phase 0 (schemas + config + Eng 6 integration).**

---

## Complete Micro-Features Catalog

Every small feature that must exist in the MASIS system, organized by subsystem. Use this as an implementation checklist.

---

### MF-SUP: Supervisor Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-SUP-01 | **First-turn DAG planning** | On `iteration_count==0`, call gpt-4.1 with `with_structured_output(TaskPlan)` to decompose query into tasks with dependencies and parallel groups | `"Compare Q3 revenue to competitors"` → `[T1(researcher) ║ T2(web_search) → T3(skeptic) → T4(synthesizer)]` |
| MF-SUP-02 | **Per-task acceptance criteria** | Each `TaskNode` has `acceptance_criteria` string that defines what "success" means. Written by the LLM during planning. | `T1.acceptance_criteria = "≥2 chunks, pass_rate≥0.30, grounded"` |
| MF-SUP-03 | **Few-shot planning prompt** | Include 4 DAG decomposition examples in the prompt (simple, comparative, multi-dimensional, thematic) so the LLM produces well-structured plans | See Q8 in Q&A section |
| MF-SUP-04 | **Fast Path: budget check** | If `token_budget.remaining <= 0` → return `"force_synthesize"`. No LLM call. | 82K/100K used → still OK. 100K/100K → force stop. |
| MF-SUP-05 | **Fast Path: iteration limit** | If `iteration_count >= MAX_SUPERVISOR_TURNS (15)` → force synthesize | Prevents runaway loops even if all other checks miss |
| MF-SUP-06 | **Fast Path: repetition detection** | Compute cosine similarity between last 2 same-type task queries. If `> 0.90` → force synthesize | `cosine("Infosys market share decline", "market share Infosys declining") = 0.91 → STOP` |
| MF-SUP-07 | **Fast Path: agent criteria check** | Parse `last_task_result` structured fields and compare against `acceptance_criteria` thresholds. No LLM needed. | `chunks_after_grading=3 >= 2 ✅ → PASS` |
| MF-SUP-08 | **Fast Path: next-task resolution** | Walk the DAG, find tasks whose dependencies are all `"done"`, return them as `next_tasks` | T1=done, T2=done → T3(depends on [T1,T2]) is now ready |
| MF-SUP-09 | **Slow Path: retry decision** | When a task fails criteria, LLM decides whether to retry with modified query | `T2.pass_rate=0.10 → LLM: "Rewrite query to be more specific"` |
| MF-SUP-10 | **Slow Path: DAG modification** | LLM can add/remove/update tasks in the remaining DAG | `"Internal docs lack competitor data" → add T2b(web_search)` |
| MF-SUP-11 | **Slow Path: HITL escalation** | When confidence is too low or risk is too high, call `interrupt()` with options | `confidence=0.38 → interrupt({options: ["expand", "accept_partial", "cancel"]})` |
| MF-SUP-12 | **Slow Path: force synthesize** | Budget/time approaching limit but enough evidence exists → skip remaining, synthesize | Budget at 90% + 3 dimensions covered out of 5 → partial answer |
| MF-SUP-13 | **Slow Path: stop** | All retries exhausted, no evidence found, user cancelled → return END | `"No evidence after 3 retries and web search. Stopping."` |
| MF-SUP-14 | **Context filtering for Supervisor** | Supervisor sees ONLY `last_task_result.summary` (200 tokens), task_dag statuses, budget. NEVER full evidence. | Prevents context bloat; Supervisor makes decisions from summaries |
| MF-SUP-15 | **DAG status tracking** | Each `TaskNode.status` updated: pending → running → done/failed. Persisted in state. | After T1 completes: `T1.status = "done"` |
| MF-SUP-16 | **Wall clock enforcement** | If `time.time() - start_time > 300s` → force synthesize | Hard timeout prevents hanging |
| MF-SUP-17 | **Supervisor decision logging** | Every decision (fast or slow) logged with reason, cost, latency for audit | `{turn: 3, mode: "fast", decision: "continue", cost: 0, latency_ms: 2}` |

---

### MF-EXE: Executor Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-EXE-01 | **Single-task dispatch** | When `len(next_tasks)==1`, call `dispatch_agent()` directly and return result | `next_tasks=[T3] → await run_skeptic(T3, state)` |
| MF-EXE-02 | **Parallel dispatch via Send()** | When `len(next_tasks)>1`, return `[Send("executor", ...) for task in next_tasks]` | `[T1, T2] → [Send("executor", {T1}), Send("executor", {T2})]` |
| MF-EXE-03 | **Agent type routing** | `dispatch_agent()` maps `task.type` to the correct Python function | `"researcher" → run_researcher()`, `"skeptic" → run_skeptic()` |
| MF-EXE-04 | **Unknown type guard** | If `task.type` not in valid types → return `AgentError` instead of crashing | `type="analyzer" → AgentError("unknown_agent_type")` |
| MF-EXE-05 | **Timeout wrapper** | Each agent call wrapped in `asyncio.wait_for(dispatch_agent(...), timeout=task_timeout)` | Researcher timeout: 30s. If exceeded → TimeoutError → Supervisor gets failure. |
| MF-EXE-06 | **Result normalization** | Every agent's output is normalized into a common `AgentOutput` that includes `task_id`, `status`, `summary` | Uniform structure for Supervisor to read regardless of agent type |
| MF-EXE-07 | **State filtering per agent** | Each agent gets a filtered view of state — not the full `MASISState` | Researcher: `{original_query, task.query}`. Skeptic: `{evidence_board, task_dag}` |
| MF-EXE-08 | **Evidence board writing** | Executor writes agent evidence to `evidence_board` via the deduplication reducer | `return {"evidence_board": result.evidence}` → reducer merges |
| MF-EXE-09 | **Budget tracking update** | After each agent call, update `token_budget.total_tokens_used` with actual tokens consumed | `budget.total_tokens_used += result.tokens_used` |
| MF-EXE-10 | **Rate limit pre-check** | Before dispatching, verify `api_call_counts[task.type] < max_total` | If researcher calls already at 8/8 → skip, return limit error |

---

### MF-VAL: Validator Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-VAL-01 | **Faithfulness scoring** | NLI-based: for each sentence in synthesis, compute entailment against source evidence | `"Revenue grew 12%" vs chunk_12 → ENTAILMENT 0.94` |
| MF-VAL-02 | **Citation accuracy check** | For each `Citation.chunk_id`, verify it exists in `evidence_board` and NLI entailment ≥ 0.80 | `Citation(chunk_id="c99")` but c99 not in evidence → FAIL |
| MF-VAL-03 | **Answer relevancy scoring** | Semantic similarity between `synthesis.answer` and `original_query` | `"Q3 revenue was ₹41,764 crore"` vs `"What was Q3 revenue?"` → 0.97 |
| MF-VAL-04 | **DAG completeness check** | % of planned subtasks addressed in the answer | DAG had 5 tasks, answer mentions 4 → completeness = 0.80 |
| MF-VAL-05 | **Threshold enforcement** | Hard gates: faithfulness≥0.85, citation_accuracy≥0.90, relevancy≥0.80, completeness≥0.90 | Any below threshold → `route_validator = "revise"` |
| MF-VAL-06 | **Score breakdown in state** | Write all scores to `quality_scores` dict for Supervisor to read on revise | `{faithfulness: 0.72, citation_accuracy: 0.91, ...}` |
| MF-VAL-07 | **Max validation rounds** | Cap validator→supervisor loops at 3 to prevent infinite revision | Round 3 fails → force END with best available |

---

### MF-RES: Researcher Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-RES-01 | **HyDE query rewrite** | Generate hypothetical answer passage to improve embedding similarity | `"Q3 revenue?" → "Infosys posted Q3 FY26 revenue of approximately..."` |
| MF-RES-02 | **Metadata extraction** | Parse year, quarter, department from query → used as ChromaDB filter | `"cloud Q3 FY26" → {year: 2026, quarter: "Q3", department: "cloud"}` |
| MF-RES-03 | **Hybrid retrieval (Vector + BM25)** | Two retrieval paths fused with RRF | Vector top 10 + BM25 top 10 → RRF(k=60) → 10 unique |
| MF-RES-04 | **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` reranks 10 chunks → keeps top 5 | Scores: [0.92, 0.87, 0.81, 0.74, 0.69, 0.45, ...] → keep first 5 |
| MF-RES-05 | **CRAG document grading** | Grade each chunk for relevance. If insufficient → rewrite and retry (max 3) | 1/5 relevant → rewrite query → retrieve → 3/5 relevant ✅ |
| MF-RES-06 | **Self-RAG hallucination check** | After generation, verify answer is grounded in evidence. If not → regenerate (max 3) | `"Revenue grew 25%"` but evidence says 12% → regenerate |
| MF-RES-07 | **Parent chunk expansion** | After reranking child chunks (500 chars), retrieve parent chunks (2000 chars) for full context | `child_42 → parent_42 (broader paragraph with surrounding context)` |
| MF-RES-08 | **Structured output** | Return `ResearcherOutput` Pydantic model with all fields Fast Path needs | `{chunks_after_grading: 3, grading_pass_rate: 0.60, self_rag_verdict: "grounded"}` |
| MF-RES-09 | **Source diversity count** | Count unique `doc_id` values in graded chunks | 3 chunks from 2 different documents → `source_diversity = 2` |
| MF-RES-10 | **200-token summary** | Truncated answer for Supervisor context (Supervisor never sees full evidence) | `"Q3 revenue ₹41,764 crore, 12% YoY growth from cloud and services"` |

---

### MF-SKE: Skeptic Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-SKE-01 | **Claim extraction** | Parse evidence board into discrete factual claims | `"Revenue grew 12% to ₹41,764 crore"` → 2 claims: "grew 12%", "₹41,764 crore" |
| MF-SKE-02 | **NLI pre-filter (BART-MNLI)** | Free, local, deterministic check of each claim vs source text | `entailment/contradiction/neutral` with confidence score |
| MF-SKE-03 | **Contradiction flagging** | If NLI label == "contradiction" and score > 0.80 → flag for LLM judge | `"improved 2.3%" vs "compressed 1.8pp" → CONTRADICTION 0.89` |
| MF-SKE-04 | **Unsupported claim flagging** | If NLI label == "neutral" and score > 0.70 → weak evidence warning | Claim with no direct evidence support |
| MF-SKE-05 | **LLM judge (o3-mini)** | Deep analysis of flagged issues + adversarial critique with min 3 issues enforced | Anti-sycophancy: different model family from Synthesizer |
| MF-SKE-06 | **Single-source detection** | Flag claims backed by only 1 chunk (no corroboration) | `"12% growth" → only chunk_12 mentions this → WARNING` |
| MF-SKE-07 | **Forward-looking statement flag** | Detect predictions used as facts | `"Expected to grow 20% next year"` → INFO: forward-looking |
| MF-SKE-08 | **Confidence scoring** | Overall confidence 0.0-1.0 based on supported vs total claims | 8/10 supported → confidence 0.80 |
| MF-SKE-09 | **Reconciliation attempt** | When contradiction detected, LLM tries to reconcile before flagging | "Different metrics" → reconciled, not contradictory |

---

### MF-SYN: Synthesizer Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-SYN-01 | **U-shape context ordering** | Reorder evidence: best→start, 2nd best→end, weakest→middle | Chunks by score [0.92, 0.87, 0.74, 0.69] → [0.92, 0.69, 0.74, 0.87] |
| MF-SYN-02 | **Critique integration** | Include Skeptic's findings in the synthesis prompt | `"Skeptic found: chunk_67 is forward-looking. Present as outlook, not fact."` |
| MF-SYN-03 | **Pydantic citation enforcement** | `citations: list[Citation] = Field(min_length=1)` makes uncited claims impossible | LLM must produce at least 1 citation or Pydantic rejects output |
| MF-SYN-04 | **Citation-claim mapping** | Each citation maps a `chunk_id` to a `claim_text` it supports | `[{chunk_id: "c12", claim_text: "₹41,764 crore"}, ...]` |
| MF-SYN-05 | **Post-hoc NLI verification** | After synthesis, NLI-score each citation's claim against its chunk | `entailment_score` field populated — used by Validator |
| MF-SYN-06 | **Partial result mode** | When `force_synthesize` is active, include disclaimer about missing aspects | `"NOTE: This covers 4 of 5 dimensions. Missing: headcount data."` |
| MF-SYN-07 | **No-evidence honest answer** | When no evidence found, say so explicitly | `"No evidence supports market share decline."` |
| MF-SYN-08 | **Both-sides presentation** | When Skeptic reconciles contradictions, present both perspectives | `"AI improved efficiency [Doc A] but compressed margins [Doc B]. Net: +0.5%"` |

---

### MF-MEM: State & Memory Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-MEM-01 | **Evidence deduplication reducer** | `Annotated[list[EvidenceChunk], evidence_reducer]` deduplicates by `(doc_id, chunk_id)`, keeps highest score | 2 researchers retrieve chunk_12 → stored once with best score |
| MF-MEM-02 | **Immutable original_query** | Never modified. All components reference back to it for drift detection. | `original_query` persists from first turn to validator |
| MF-MEM-03 | **Iteration counter** | Global `iteration_count` incremented every executor turn. Used by Fast Path. | `iteration_count: 7` → still under 15 limit |
| MF-MEM-04 | **Budget tracker** | `BudgetTracker` tracks tokens used, cost USD, API calls per agent type | `{total_tokens: 45000, cost: 0.23, calls: {"researcher": 3, "skeptic": 1}}` |
| MF-MEM-05 | **Task status in DAG** | Each `TaskNode.status` updated as execution progresses | `T1.status: "pending" → "running" → "done"` |
| MF-MEM-06 | **Whiteboard pattern** | `evidence_board` = shared memory. Agents write to it, read from it. No direct agent-to-agent communication. | T1 writes chunks → T3(skeptic) reads all chunks |
| MF-MEM-07 | **Filtered state views** | Each agent gets a subset of state. Prevents leaking irrelevant info. | Researcher: `{task.query}`. Supervisor: `{summary, budget}`. Skeptic: `{all evidence}`. |
| MF-MEM-08 | **Checkpoint persistence** | PostgresSaver saves state after every super-step. Enables HITL resume, fault tolerance, audit. | Kill process → restart → `thread_id` → exact state restored |

---

### MF-HITL: Human-in-the-Loop Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-HITL-01 | **Pre-supervisor ambiguity gate** | Classify query before DAG planning. AMBIGUOUS → interrupt with options. | `"How is the tech division?"` → options: Cloud / AI / Enterprise / All |
| MF-HITL-02 | **DAG approval pause** | After Supervisor plans DAG, interrupt for user approval/editing | User can add tasks, remove tasks, modify queries |
| MF-HITL-03 | **Mid-execution evidence pause** | When coverage < threshold, pause with partial results and options | `"Found 4/5 dimensions. Missing: headcount. Accept partial?"` |
| MF-HITL-04 | **Risk gate pause** | For financial/legal recommendations, require human sign-off | `risk_score > 0.7 → "This involves investment recommendations. Please review."` |
| MF-HITL-05 | **Resume with action** | `Command(resume={"action": "expand_to_web"})` continues pipeline with user's choice | Graph resumes from exact checkpoint, Supervisor reads resume value |
| MF-HITL-06 | **Graceful partial result** | When user accepts partial, synthesize with `missing_aspects` disclaimer | `PartialResult(coverage=0.80, missing=["headcount"], disclaimer="...")` |
| MF-HITL-07 | **Cancel support** | User can cancel → END with explanation of work done so far | `{status: "cancelled", work_completed: ["T1", "T2"], evidence_found: [...]}` |

---

### MF-SAFE: Safety & Resilience Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-SAFE-01 | **3-layer loop prevention** | CRAG (3 retries) + cosine (>0.90) + hard cap (15 turns) | No path to infinite loop |
| MF-SAFE-02 | **Circuit breaker (3-state)** | CLOSED → OPEN (4 failures) → HALF-OPEN (probe@60s) → CLOSED | API failure → automatic fallback → automatic recovery |
| MF-SAFE-03 | **Model fallback chains** | Per-role: Primary → Fallback → Last Resort | Researcher: gpt-4.1-mini → gpt-4.1 → return partial |
| MF-SAFE-04 | **Content sanitizer** | Strip prompt injection patterns from web search results before LLM context | `"IGNORE PREVIOUS INSTRUCTIONS"` → stripped |
| MF-SAFE-05 | **Rate limiting per agent** | `max_parallel`, `max_total`, `timeout_s` per agent type | Researcher: max 3 parallel, 8 total, 30s timeout |
| MF-SAFE-06 | **Budget enforcement** | Hard caps: 100K tokens, $0.50, 300s wall clock per query | Approaching limit → force synthesize with best evidence |
| MF-SAFE-07 | **Graceful degradation** | Never crash. Always produce best-effort answer with transparency about limitations. | Budget exhausted → partial answer + disclaimer |
| MF-SAFE-08 | **Drift detection** | `answer_relevancy(answer, original_query) >= 0.80` in Validator | Answer drifts from query → revise loop |

---

### MF-API: API & Observability Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-API-01 | **POST /query** | Accept query, create thread_id, start graph async | Returns `{thread_id: "abc", status: "processing"}` |
| MF-API-02 | **POST /resume** | Accept resume action, call `graph.ainvoke(Command(resume=...))` | HITL resume: `{action: "expand_to_web"}` |
| MF-API-03 | **GET /status** | Return current state: processing/paused/completed/failed | Quick polling for frontend |
| MF-API-04 | **GET /trace** | Return full audit trail: DAG, decisions, evidence, quality scores | Full transparency for compliance |
| MF-API-05 | **GET /stream** | SSE stream of typed events during execution | `plan_created`, `task_started`, `task_completed`, `hitl_required`, `answer_ready` |
| MF-API-06 | **Langfuse/LangSmith tracing** | Every LLM call traced with inputs/outputs, latency, cost | Debug production issues, optimize prompts |
| MF-API-07 | **Prometheus metrics** | `query_latency_seconds`, `cost_per_query_usd`, `agent_call_count`, `fast_path_ratio` | Grafana dashboards |
| MF-API-08 | **Model routing config** | Centralized `config/model_routing.py` with env var overrides | `MODEL_RESEARCHER=gpt-4.1-nano` → all researcher calls use nano |

---

### MF-EVAL: Evaluation Micro-Features

| ID | Feature | What It Does | Example |
|---|---|---|---|
| MF-EVAL-01 | **Golden dataset** | 50+ curated queries spanning all query types | Simple, comparative, contradictory, ambiguous, thematic |
| MF-EVAL-02 | **Regression runner** | Weekly automated run against golden dataset; fail if metrics drop | `faithfulness` drops from 0.91 to 0.83 → alert |
| MF-EVAL-03 | **Per-scenario testing** | Each of the 10 simulation scenarios as an E2E test | S1-S10 with expected outcome assertions |
| MF-EVAL-04 | **Cost tracking per query** | Log total cost + breakdown per agent per query | Detect cost regressions from prompt changes |

---

### Total: 85 Micro-Features across 9 Subsystems

| Subsystem | Count |
|---|---|
| MF-SUP: Supervisor | 17 |
| MF-EXE: Executor | 10 |
| MF-VAL: Validator | 7 |
| MF-RES: Researcher | 10 |
| MF-SKE: Skeptic | 9 |
| MF-SYN: Synthesizer | 8 |
| MF-MEM: State & Memory | 8 |
| MF-HITL: Human-in-the-Loop | 7 |
| MF-SAFE: Safety & Resilience | 8 |
| MF-API: API & Observability | 8 |
| MF-EVAL: Evaluation | 4 |
| **TOTAL** | **96** |


