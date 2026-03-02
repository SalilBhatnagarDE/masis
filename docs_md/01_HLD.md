# MASIS — System Overview

```mermaid
flowchart TD

    Q(["👤 User Query"])

    Q -->|"ambiguous → HITL\nout-of-scope → reject\nclear ↓"| SUP

    subgraph SUP_BOX["━━━━━━━━━━━━  🧠 SUPERVISOR — The Brain  ━━━━━━━━━━━━"]
        SUP["MODE 1  ·  plan_dag\n──────────────────────────────────────────────\nFewShot + gpt-4.1  →  with_structured_output(TaskPlan)\nDecomposes query into a typed Task DAG\nSelects which agents run · in what order · in parallel\n══════════════════════════════════════════════\nMODE 2  ·  Fast Path monitor  ( $0 · <10 ms )\n──────────────────────────────────────────────\nChecks every turn: budget · iteration cap · cosine loop detect\nReads per-task acceptance criteria from each agent result\nIf criteria PASS  →  dispatch next tasks\nIf budget / cap hit  →  force_synthesize\n══════════════════════════════════════════════\nMODE 3  ·  Slow Path LLM  ( gpt-4.1 · ~$0.015 )\n──────────────────────────────────────────────\nOnly fires when Fast Path cannot decide\nCan:  🔄 retry  ·  📋 re-plan DAG  ·  ⏹ stop  ·  🔔 escalate to human"]
    end

    SUP_BOX -->|"① dispatch\ntasks as DAG"| EXE_BOX

    subgraph EXE_BOX["━━━━━━━━━━  ⚙️ EXECUTOR — Dynamic DAG Runner  ━━━━━━━━━━"]

        subgraph POOL["📦  Agent Pool  —  Supervisor picks, sequences, and parallelises these"]
            direction LR
            AG1["🔬 Researcher A\nSemantic RAG\nHyDE · CRAG · Self-RAG\nHybrid BM25 + vector"]
            AG2["🌐 Researcher B\nWeb Search\nTavily live data\ninjection-sanitized"]
            AG3["🔍 Skeptic\nNLI pre-filter\nAdversarial LLM judge\nHallucination audit"]
            AG4["✍️ Synthesizer\nU-shape ordering\nPydantic citations\nFinal draft"]
        end

        subgraph DAG_EX["🔀  Example DAG for: 'Compare Q3 cloud revenue'"]
            direction LR
            T1["Researcher A\nsemantic docs"] & T2["Researcher B\nweb live data"] --> T3["Skeptic\ncross-check"] --> T4["Synthesizer\ncited answer"]
        end

    end

    EXE_BOX -->|"② task results\n+ typed AgentOutput"| VAL_BOX

    subgraph VAL_BOX["━━━━━━━━━━━━  ✅ VALIDATOR — Quality Gate  ━━━━━━━━━━━━"]
        VAL["Faithfulness ≥ 0.85     ·     Citation Accuracy ≥ 0.90\nAnswer Relevancy ≥ 0.80     ·     DAG Completeness ≥ 0.90\n─────────────────────────────────────────────────────\nAll pass  →  Final Answer          Any fail  →  back to Supervisor"]
    end

    VAL_BOX -->|"③ fail\nrevise loop"| SUP_BOX
    VAL_BOX -->|"✅ pass"| ANS(["✅ Final Answer\nCited · Validated · Safe"])

    EXE_BOX -.->|"per-task acceptance\ncriteria results"| SUP_BOX

    SUP_BOX -.->|"🔄 retry\nfailed task"| EXE_BOX
    SUP_BOX -.->|"📋 re-plan\nmodify DAG"| EXE_BOX
    SUP_BOX -.->|"⏹ budget cap\nforce stop"| VAL_BOX
    SUP_BOX -.->|"🔔 escalate\nto human"| HITL(["👤 Human Review"])

    style SUP_BOX fill:#0d2137,stroke:#4A90D9,stroke-width:2px,color:#cce4ff
    style EXE_BOX fill:#0d1a0d,stroke:#27AE60,stroke-width:2px,color:#ccffcc
    style POOL  fill:#112211,stroke:#2ECC71,color:#b8ffb8
    style DAG_EX fill:#001a00,stroke:#1abc9c,color:#b8ffee
    style VAL_BOX fill:#1a0d2e,stroke:#9B59B6,stroke-width:2px,color:#e8d5ff
    style ANS fill:#1a5c1a,stroke:#27AE60,color:white
    style HITL fill:#5c1a00,stroke:#E74C3C,color:#ffd5cc
    style T1 fill:#1a3a5c,stroke:#4A90D9,color:#cce4ff
    style T2 fill:#3a2200,stroke:#E67E22,color:#ffe8cc
    style T3 fill:#3a0000,stroke:#E74C3C,color:#ffd5cc
    style T4 fill:#003a00,stroke:#27AE60,color:#ccffcc
    style AG1 fill:#1a3a5c,stroke:#4A90D9,color:#cce4ff
    style AG2 fill:#3a2200,stroke:#E67E22,color:#ffe8cc
    style AG3 fill:#3a0000,stroke:#E74C3C,color:#ffd5cc
    style AG4 fill:#003a00,stroke:#27AE60,color:#ccffcc
```

> **How to present this:** Start at the top — "the user's query enters and the Supervisor immediately decides if it's even worth processing." Then walk the happy path (plan → dispatch → execute DAG → validate → answer). Then point to the dotted lines — "but the Supervisor never lets go of control: it reads every task result, and can retry, re-plan, force-stop, or escalate to a human at any turn." Finish with the validator loop — "and even after synthesis, the answer must pass four quality gates before the user sees it."

---

# Slide 1: High-Level Design -- "The Brain Trust"

> **Pillar:** High-Level Design (HLD)
> **Time Allocation:** 4-5 minutes
> **Curveball Addressed:** "Why only 3 graph nodes instead of one per agent?" / "Can you explain this to a non-technical CTO?"

---

## System Overview

MASIS (Multi-Agent Supervised Intelligence System) is a **3-node LangGraph StateGraph** that orchestrates a dynamic task DAG for enterprise document research. The graph is the engine; the DAG is the fuel.

### Core Principle: Separation of Concerns

| Concept | What It Is | When Created | Who Creates It |
|---------|-----------|--------------|----------------|
| **LangGraph Execution Graph** | Fixed 3-node structure: Supervisor - Executor - Validator. Built once at startup. Never changes. | `workflow.compile()` at startup | Developer |
| **Task DAG** | Dynamic research plan: `T1(researcher) \|\| T2(web_search) -> T3(skeptic) -> T4(synthesizer)`. Data inside state. | Runtime, Supervisor's first turn | Supervisor LLM (gpt-4.1) |

---

## The 3-Node Architecture

```mermaid
graph TD
    START([START]) --> SUP[Supervisor Node<br/><i>The Brain</i><br/>Plan / Fast Path / Slow Path]

    SUP -->|"continue /<br/>force_synthesize"| EXE[Executor Node<br/><i>The Dispatcher</i><br/>Routes to agent functions]
    SUP -->|"ready_for_validation"| VAL[Validator Node<br/><i>The Quality Gate</i><br/>faithfulness / citation / relevancy]
    SUP -->|"hitl_pause / failed"| ENDNODE([END])

    EXE -->|"always returns to"| SUP

    VAL -->|"pass"| ENDNODE
    VAL -->|"revise"| SUP

    style SUP fill:#4A90D9,stroke:#2C5F8A,color:white
    style EXE fill:#7B68EE,stroke:#5A4BB6,color:white
    style VAL fill:#2ECC71,stroke:#27AE60,color:white
    style ENDNODE fill:#95A5A6,stroke:#7F8C8D,color:white
    style START fill:#95A5A6,stroke:#7F8C8D,color:white
```

**Three nodes. Two loops. The Supervisor is always in control.**

---

## Why Only 3 Graph Nodes?

The agents (Researcher, Skeptic, Synthesizer, Web Search) are **Python functions called by the Executor** -- NOT separate LangGraph nodes.

```python
# Inside executor -- these are just function calls, NOT graph nodes:
async def dispatch_agent(task: TaskNode, state: MASISState):
    if task.type == "researcher":    return await run_researcher(task, state)
    if task.type == "web_search":    return await run_web_search(task)
    if task.type == "skeptic":       return await run_skeptic(task, state)
    if task.type == "synthesizer":   return await run_synthesizer(task, state)
```

| 3-Node Advantage | Separate-Agent-Node Disadvantage |
|-------------------|-----------------------------------|
| Simple graph -- 3 nodes, easy to reason about | 5+ conditional edges, complex routing |
| Supervisor always sees results between tasks | Supervision gaps between graph steps |
| Easy to add new agent types -- just a new `if` | Need to rewire graph edges |
| DAG drives dispatch -- structural enforcement | LLM might skip the Skeptic |

> **Codebase Reference:** `C:\Users\salil\final_maiss\masis\graph\workflow.py` -- `build_workflow()` function (lines 88-156)

---

## The Dynamic Task DAG

The Supervisor creates a DAG on the first turn using gpt-4.1 with `with_structured_output(TaskPlan)`. Each task has per-task acceptance criteria written in natural language by the LLM.

```mermaid
graph LR
    subgraph "Parallel Group 1"
        T1["T1: Researcher<br/>TechCorp cloud revenue"]
        T2["T2: Web Search<br/>AWS Azure GCP revenue"]
        T3["T3: Researcher<br/>Growth trend 5 years"]
    end

    subgraph "Group 2"
        T4["T4: Skeptic<br/>Cross-check evidence"]
    end

    subgraph "Group 3"
        T5["T5: Synthesizer<br/>Comparison + trends"]
    end

    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> T5

    style T1 fill:#3498DB,color:white
    style T2 fill:#E67E22,color:white
    style T3 fill:#3498DB,color:white
    style T4 fill:#E74C3C,color:white
    style T5 fill:#2ECC71,color:white
```

---

## Supervisor Two-Tier Decision System

The Supervisor uses a **Fast Path / Slow Path** split. In practice, **60-70% of Supervisor runs are Fast Path** -- free and instant.

```mermaid
flowchart TD
    ENTRY[Supervisor Called] --> CHECK_ITER{iteration == 0?}
    CHECK_ITER -->|Yes| PLAN["MODE 1: PLAN<br/>(gpt-4.1, ~$0.012)<br/>Decompose query -> DAG"]
    CHECK_ITER -->|No| FAST["MODE 2: FAST PATH<br/>(Rules, $0, <10ms)"]

    FAST --> FP1{Budget exhausted?}
    FP1 -->|Yes| FORCE[force_synthesize]
    FP1 -->|No| FP2{Iteration >= 15?}
    FP2 -->|Yes| FORCE
    FP2 -->|No| FP3{Wall clock > 300s?}
    FP3 -->|Yes| FORCE
    FP3 -->|No| FP4{Repetitive search?<br/>cosine > 0.90}
    FP4 -->|Yes| FORCE
    FP4 -->|No| FP5{Criteria PASS?}
    FP5 -->|Yes| NEXT[Dispatch next<br/>ready tasks]
    FP5 -->|No| SLOW["MODE 3: SLOW PATH<br/>(gpt-4.1, ~$0.015)"]

    SLOW --> S1[retry]
    SLOW --> S2[modify_dag]
    SLOW --> S3[escalate / HITL]
    SLOW --> S4[force_synthesize]
    SLOW --> S5[stop]

    style PLAN fill:#E74C3C,color:white
    style FAST fill:#2ECC71,color:white
    style SLOW fill:#E74C3C,color:white
    style FORCE fill:#F39C12,color:white
    style NEXT fill:#3498DB,color:white
```

> **Codebase Reference:** `C:\Users\salil\final_maiss\masis\nodes\supervisor.py` -- `monitor_and_route()` (lines 433-532)

---

## 96 Micro-Features Across 11 Subsystems

| Subsystem | Count | Key Features |
|-----------|-------|-------------|
| MF-SUP: Supervisor | 17 | DAG planning, Fast/Slow Path, HITL escalation, decision logging |
| MF-EXE: Executor | 10 | Send() parallel dispatch, timeout wrapper, rate limiting |
| MF-VAL: Validator | 7 | Faithfulness, citation accuracy, relevancy, completeness gates |
| MF-RES: Researcher | 10 | HyDE, hybrid retrieval, CRAG, Self-RAG, parent expansion |
| MF-SKE: Skeptic | 9 | NLI pre-filter, LLM judge, contradiction reconciliation |
| MF-SYN: Synthesizer | 8 | U-shape ordering, Pydantic citations, partial result mode |
| MF-MEM: State & Memory | 8 | Evidence reducer, immutable query, filtered views, checkpoints |
| MF-HITL: Human-in-the-Loop | 7 | Ambiguity gate, DAG approval, risk gate, cancel support |
| MF-SAFE: Safety | 8 | 3-layer loop prevention, circuit breaker, model fallback |
| MF-API: API & Observability | 8 | 5 REST endpoints, SSE streaming, Prometheus metrics |
| MF-EVAL: Evaluation | 4 | Golden dataset, regression runner, per-scenario testing |
| **TOTAL** | **96** | |

---

## Key Design Decisions

1. **3-node graph, not N-agent graph:** Keeps routing simple, guarantees supervision after every task
2. **DAG as data, not as graph structure:** Enables dynamic modification at runtime (add/remove tasks)
3. **Two-tier Supervisor:** Fast Path eliminates 60-70% of LLM calls, saving ~$0.10/query
4. **Agents as functions, not nodes:** Adding a new agent type = adding one `if` branch
5. **Parallel execution via Send():** LangGraph's `Send()` dispatches independent tasks concurrently

---

## Presenter Talking Points

1. "MASIS uses a 3-node LangGraph StateGraph -- Supervisor, Executor, Validator -- with the task DAG stored as data in state, not as graph topology."

2. "The Supervisor operates in three modes: PLAN on first turn, FAST PATH for 60-70% of subsequent turns at zero cost, and SLOW PATH only when tasks fail criteria."

3. "Agents are Python functions called by the Executor, not separate graph nodes. This means adding a new agent is a one-line change, and the Supervisor always sees results between every single task."

4. "The system implements 96 micro-features across 11 subsystems, with every feature independently testable and traceable to a micro-feature ID."

---

> **Wow Statement:** "We turned a complex multi-agent system into just three graph nodes -- because the smartest architecture is the one that is simple enough to debug at 3 AM."
