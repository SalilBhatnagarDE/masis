# Low-Level Design

This document covers the internal data structures, state schema, agent dispatch flow, and routing logic that make MASIS work.

---

## MASISState: The Shared Whiteboard

Everything flows through a single `MASISState` TypedDict. LangGraph nodes return partial dicts — only the keys they modify. Agents don't talk to each other directly; they read from and write to this shared state.

```python
# masis/schemas/models.py

class MASISState(TypedDict, total=False):
    # Immutable query identity
    original_query: str                              # NEVER modified after first turn
    query_id: str                                    # UUID for checkpoint + tracing

    # Supervisor-owned
    task_dag: List[TaskNode]                          # Dynamic DAG
    stop_condition: str                              # When is the query "done"?
    iteration_count: int                             # Global counter
    next_tasks: List[TaskNode]                       # Tasks to dispatch next
    supervisor_decision: str                         # Routing output for edges
    last_task_result: Optional[AgentOutput]           # Most recent agent output

    # Evidence whiteboard (parallel-safe via custom reducer)
    evidence_board: Annotated[List[EvidenceChunk], evidence_reducer]

    # Agent results
    critique_notes: Optional[SkepticOutput]           # Skeptic findings
    synthesis_output: Optional[SynthesizerOutput]     # Final answer

    # Quality & validation
    quality_scores: Dict[str, float]
    validation_round: int                            # Capped at 2

    # Budget & safety
    token_budget: BudgetTracker                      # 200K tokens / $0.50 / 300s
    api_call_counts: Dict[str, int]                  # Per-agent rate limiting

    # Audit
    decision_log: List[Dict[str, Any]]               # Every Supervisor decision
```

---

## Evidence Deduplication Reducer

When two parallel Researchers retrieve the same chunk, the reducer keeps only the copy with the highest retrieval score. This is registered via Python's `Annotated` type — LangGraph calls it automatically on every state update.

```python
# masis/schemas/models.py

def evidence_reducer(
    existing: List[EvidenceChunk],
    new: List[EvidenceChunk],
) -> List[EvidenceChunk]:
    """Dedup by (doc_id, chunk_id). Keep highest score."""
    index: Dict[tuple, EvidenceChunk] = {}
    for chunk in existing:
        key = (chunk.doc_id, chunk.chunk_id)
        index[key] = chunk
    for chunk in new:
        key = (chunk.doc_id, chunk.chunk_id)
        if key not in index or chunk.retrieval_score > index[key].retrieval_score:
            index[key] = chunk
    return list(index.values())

# Usage in state:
evidence_board: Annotated[List[EvidenceChunk], evidence_reducer]
# Agents just return: {"evidence_board": result.evidence}
# The reducer merges — it does NOT replace.
```

State grows linearly with unique evidence, not with task count. No locks needed.

---

## Execution Flow (Sequence Diagram)

```mermaid
sequenceDiagram
    participant U as User
    participant SUP as Supervisor
    participant EXE as Executor
    participant VAL as Validator
    participant STATE as MASISState

    U->>SUP: Raw Query
    Note over SUP: MODE 1: PLAN (gpt-4.1)
    SUP->>STATE: Write: task_dag, stop_condition
    SUP->>EXE: next_tasks = [T1, T2] (parallel)

    par Parallel via Send()
        EXE->>STATE: T1: Write evidence_board (via reducer)
    and
        EXE->>STATE: T2: Write evidence_board (via reducer)
    end

    EXE->>SUP: last_task_result (AgentOutput)
    Note over SUP: MODE 2: FAST PATH ($0, <10ms)
    SUP->>SUP: Check criteria: PASS
    SUP->>EXE: next_tasks = [T3 skeptic]

    EXE->>STATE: T3: Read evidence_board, Write critique_notes
    EXE->>SUP: last_task_result

    Note over SUP: MODE 2: FAST PATH
    SUP->>EXE: next_tasks = [T4 synthesizer]

    EXE->>STATE: T4: Read evidence + critique, Write synthesis_output
    EXE->>SUP: last_task_result

    Note over SUP: All done → ready_for_validation
    SUP->>VAL: Route to Validator

    VAL->>STATE: Read synthesis, evidence
    VAL->>VAL: Score: faithfulness, citation, relevancy, completeness

    alt All scores pass thresholds
        VAL->>U: Final Answer + Audit Trail
    else Any score below threshold
        VAL->>SUP: revise (quality_scores in state)
        Note over SUP: MODE 3: SLOW PATH (gpt-4.1)
        SUP->>EXE: Retry failed dimension
    end
```

---

## Filtered State Views Per Agent

Each agent sees only what it needs. The Supervisor never sees full evidence chunks.

| Agent | Sees | Does NOT See |
|---|---|---|
| **Supervisor** | `original_query`, `task_dag` (statuses), `last_task_result.summary` (500 chars max), `token_budget`, `iteration_count` | Full `evidence_board`, `critique_notes`, `synthesis_output` |
| **Researcher** | `task.query` (its own sub-question only) | Other researchers' evidence, `task_dag`, budget |
| **Skeptic** | All `evidence_board` chunks (needs cross-doc analysis), `task_dag` | Budget, `iteration_count`, other agent summaries |
| **Synthesizer** | `evidence_board` (U-shape ordered), `critique_notes`, `task_dag` | Budget, raw retrieval scores |
| **Validator** | `synthesis_output`, `evidence_board`, `original_query`, `task_dag` | Budget, `decision_log` |

The Supervisor makes routing decisions from 500-char summaries, not 100,000 chars of raw evidence. This keeps its context window small and its decisions fast.

---

## Pydantic Schema Hierarchy

```mermaid
classDiagram
    class MASISState {
        +str original_query
        +str query_id
        +List~TaskNode~ task_dag
        +int iteration_count
        +Annotated~List~EvidenceChunk~~ evidence_board
        +BudgetTracker token_budget
        +Dict quality_scores
    }

    class TaskNode {
        +str task_id
        +Literal type
        +str query
        +List~str~ dependencies
        +int parallel_group
        +str acceptance_criteria
        +Literal status
        +bool is_ready(done_ids)
    }

    class TaskPlan {
        +List~TaskNode~ tasks
        +str stop_condition
        +last_group_must_have_synthesizer()
    }

    class AgentOutput {
        +str task_id
        +str agent_type
        +Literal status
        +str summary [max 500 chars]
        +List~EvidenceChunk~ evidence
        +Dict criteria_result
        +int tokens_used
    }

    class EvidenceChunk {
        +str chunk_id
        +str doc_id
        +str parent_chunk_id
        +str text
        +float retrieval_score
        +float rerank_score
        +Dict metadata
        +tuple dedup_key()
    }

    class ResearcherOutput {
        +int chunks_after_grading
        +float grading_pass_rate
        +Literal self_rag_verdict
        +int source_diversity
        +Dict to_criteria_dict()
    }

    class SkepticOutput {
        +int claims_checked
        +int claims_supported
        +int claims_contradicted
        +float overall_confidence
        +List reconciliations
        +List forward_looking_flags
    }

    class SynthesizerOutput {
        +str answer
        +List~Citation~ citations [min=1]
        +bool is_partial
        +List missing_dimensions
    }

    class Citation {
        +str chunk_id
        +str claim_text
        +float entailment_score
    }

    class BudgetTracker {
        +int total_tokens_used
        +float total_cost_usd
        +int remaining [default=200000]
        +Dict api_calls
        +float start_time
        +bool is_exhausted()
        +BudgetTracker add(tokens, cost)
    }

    MASISState --> TaskNode
    MASISState --> EvidenceChunk
    MASISState --> AgentOutput
    MASISState --> BudgetTracker
    MASISState --> SkepticOutput
    MASISState --> SynthesizerOutput
    TaskPlan --> TaskNode
    SynthesizerOutput --> Citation
    AgentOutput --> EvidenceChunk
    AgentOutput --> ResearcherOutput : raw_output
```

---

## Routing Logic

Routing is pure Python — reads one string field, returns next node name. No LLM involved.

```python
# masis/graph/edges.py

def route_supervisor(state: MASISState) -> str:
    decision = state.get("supervisor_decision", "failed")

    if decision == "continue":             return "executor"
    if decision == "ready_for_validation": return "validator"
    if decision == "force_synthesize":     return "executor"  # executor checks flag
    if decision == "hitl_pause":           return END
    if decision == "failed":               return END
    return END

def route_validator(state: MASISState) -> str:
    if state.get("validation_pass", False): return END
    return "supervisor"
```

| Decision | Next Node | When |
|---|---|---|
| `"continue"` | Executor | Normal task dispatch |
| `"ready_for_validation"` | Validator | All DAG tasks complete |
| `"force_synthesize"` | Executor | Budget/time/repetition cap hit |
| `"hitl_pause"` | END | Human review needed |
| `"failed"` | END | Unrecoverable error |
| Validator `pass` | END | All quality gates met |
| Validator `revise` | Supervisor | At least one threshold missed |

---

## Tool Exposure Per Agent

| Capability | Supervisor | Researcher | Skeptic | Synthesizer | Validator |
|---|:---:|:---:|:---:|:---:|:---:|
| gpt-4.1 | Yes | — | — | Yes | — |
| gpt-4.1-mini | — | Yes | — | — | — |
| o3-mini | — | — | Yes | — | — |
| ChromaDB vector search | — | Yes | — | — | — |
| BM25 keyword search | — | Yes | — | — | — |
| Cross-encoder reranking | — | Yes | — | — | — |
| BART-MNLI (NLI) | — | — | Yes | Yes (post-hoc) | Yes |
| Tavily web search | — | Yes (via task type) | — | — | — |
| `interrupt()` (HITL) | Yes | — | — | — | — |
| Pydantic structured output | Yes | Yes | Yes | Yes | — |

---

## Graph Wiring

The 3-node StateGraph is built once at startup. The task DAG is data inside state, not topology.

```python
# masis/graph/workflow.py

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

workflow = StateGraph(MASISState)

# 3 nodes only
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("executor", execute_dag_tasks)
workflow.add_node("validator", final_validation)

# Edges
workflow.set_entry_point("supervisor")
workflow.add_edge("executor", "supervisor")   # executor always returns to supervisor

workflow.add_conditional_edges("supervisor", route_supervisor, {
    "continue":             "executor",
    "ready_for_validation": "validator",
    "force_synthesize":     "executor",
    "hitl_pause":           END,
    "failed":               END,
})

workflow.add_conditional_edges("validator", route_validator, {
    "pass":   END,
    "revise": "supervisor",
})

checkpointer = InMemorySaver()   # Dev; switch to PostgresSaver for production
graph = workflow.compile(checkpointer=checkpointer)
```

---

## Supervisor Node — Three Modes

```python
# masis/nodes/supervisor.py

async def supervisor_node(state: MASISState) -> dict:

    # MODE 1: PLAN — first turn only, always Slow Path
    if state["iteration_count"] == 0:
        return await plan_dag(state)

    last_result = state["last_task_result"]

    # MODE 2: FAST PATH — rule-based, no LLM, $0, <10ms
    if state["token_budget"].remaining <= 0:
        return {"supervisor_decision": "force_synthesize", "reason": "budget_exhausted"}

    if state["iteration_count"] >= MAX_SUPERVISOR_TURNS:   # 15
        return {"supervisor_decision": "force_synthesize", "reason": "max_iterations"}

    if is_repetitive(state):
        return {"supervisor_decision": "force_synthesize", "reason": "repetitive_loop"}

    criteria_result = check_agent_criteria(last_result)
    if criteria_result == "PASS":
        next_tasks = get_next_ready_tasks(state["task_dag"])
        if not next_tasks:
            return {"supervisor_decision": "ready_for_validation"}
        return {"supervisor_decision": "continue", "next_tasks": next_tasks,
                "iteration_count": state["iteration_count"] + 1}

    # MODE 3: SLOW PATH — LLM needed (~$0.015, ~3s)
    return await supervisor_llm_decision(state)
```

---

## Per-Agent Fast Path Criteria

The Fast Path checks these rules without calling an LLM. Failure triggers the Slow Path.

| Agent | PASS conditions | FAIL triggers |
|---|---|---|
| **Researcher** | `chunks_after_grading >= 2` AND `grading_pass_rate >= 0.30` AND `self_rag_verdict == "grounded"` | Any threshold missed |
| **Skeptic** | `claims_unsupported == 0` AND `claims_contradicted == 0` AND `len(logical_gaps) == 0` AND `confidence >= 0.65` | Any flag raised |
| **Synthesizer** | `citations_count >= claims_count` AND `all_citations_valid == True` | Missing or invalid citations |
| **Web Search** | `relevant_results >= 1` AND `timeout == False` | Empty result or timeout |

Slow Path decision options:

| Decision | Example trigger | What happens |
|---|---|---|
| **Retry** | Researcher pass_rate=0.10 | Re-dispatch with rewritten query |
| **Modify DAG** | Internal docs lack competitor data | Add web_search task, update deps |
| **Escalate (HITL)** | Skeptic contradiction confidence=0.38 | `interrupt()` → user decides |
| **Force synthesize** | Budget at 90%, enough for partial answer | Skip remaining, synthesize with caveat |
| **Stop** | All retries exhausted | Return END with stop_reason |

---

## Executor Node — Parallel Dispatch

```python
# masis/nodes/executor.py

async def execute_dag_tasks(state: MASISState) -> dict:
    next_tasks = state["next_tasks"]

    if len(next_tasks) == 1:
        result = await dispatch_agent(next_tasks[0], state)
        return {
            "last_task_result": result,
            "evidence_board": result.evidence if hasattr(result, 'evidence') else [],
            "iteration_count": state["iteration_count"] + 1,
        }
    else:
        # Parallel: LangGraph Send() fans out multiple tasks
        return [
            Send("executor", {"next_tasks": [task], **filtered_state(state, task)})
            for task in next_tasks
        ]

async def dispatch_agent(task: TaskNode, state: MASISState):
    if task.type == "researcher":    return await run_researcher(task, state)
    elif task.type == "web_search":  return await run_web_search(task)
    elif task.type == "skeptic":     return await run_skeptic(task, state)
    elif task.type == "synthesizer": return await run_synthesizer(task, state)
```

Each parallel task in the same `parallel_group` is dispatched via `Send()`. LangGraph runs them concurrently. Both write to `evidence_board` through the `evidence_reducer`.

---

## Supervisor Internals

### DAG Planning — First Turn

```python
async def plan_dag(state: MASISState) -> dict:
    llm = ChatOpenAI(model=MODEL_ROUTING["supervisor_plan"], temperature=0.2)
    structured_llm = llm.with_structured_output(TaskPlan)

    plan: TaskPlan = await structured_llm.ainvoke([
        SystemMessage(content=SUPERVISOR_PLAN_PROMPT),  # 4 few-shot examples
        HumanMessage(content=f"Query: {state['original_query']}")
    ])

    return {
        "task_dag": plan.tasks,
        "stop_condition": plan.stop_condition,
        "supervisor_decision": "continue",
        "next_tasks": get_next_ready_tasks(plan.tasks),
        "iteration_count": 1,
    }
```

The planning prompt includes 4 few-shot examples covering simple factual, comparative (needs web), multi-dimensional, and thematic queries. Each example shows correct `parallel_group` usage and measurable `acceptance_criteria`.

### Fast Path: Criteria Checking

```python
def check_agent_criteria(task: TaskNode, result: AgentOutput) -> str:
    if task.type == "researcher":
        checks = {
            "chunks":    result.chunks_after_grading >= 2,
            "pass_rate": result.grading_pass_rate >= 0.30,
            "grounding": result.self_rag_verdict == "grounded",
        }
    elif task.type == "skeptic":
        checks = {
            "unsupported":  result.claims_unsupported <= 0,
            "contradicted": result.claims_contradicted <= 0,
            "gaps":         len(result.logical_gaps) <= 0,
            "confidence":   result.overall_confidence >= 0.65,
        }
    elif task.type == "synthesizer":
        checks = {
            "citations_exist": result.citations_count >= result.claims_count,
            "citations_valid": result.all_citations_in_evidence_board,
        }
    elif task.type == "web_search":
        checks = {
            "results":    result.relevant_results >= 1,
            "no_timeout": not result.timeout,
        }

    return "PASS" if all(checks.values()) else "FAIL"
```

### Fast Path: DAG Walking

```python
def get_next_ready_tasks(dag: list[TaskNode]) -> list[TaskNode]:
    done_ids = {t.task_id for t in dag if t.status == "done"}

    ready = [
        t for t in dag
        if t.status == "pending"
        and all(dep in done_ids for dep in t.dependencies)
    ]

    if not ready:
        return []

    # All tasks in the lowest group number dispatch together (they're parallel)
    min_group = min(t.parallel_group for t in ready)
    return [t for t in ready if t.parallel_group == min_group]
```

Example trace:
```
DAG: T1(group=1, done) ║ T2(group=1, done) → T3(group=2, pending) → T4(group=3, pending)

done_ids = {"T1", "T2"}
T3: pending AND deps [T1,T2] all done → READY
T4: pending BUT dep T3 not done → NOT ready

Return: [T3]  → Executor dispatches T3
```

### Fast Path: Repetition Detection

```python
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # local, free

def is_repetitive(state: MASISState) -> bool:
    dag = state["task_dag"]
    last_type = state["last_task_result"].task_type

    same_type = [t for t in dag if t.type == last_type and t.status in ("done", "failed")]
    if len(same_type) < 2:
        return False

    last_two = same_type[-2:]
    emb1 = embedder.encode(last_two[0].query)
    emb2 = embedder.encode(last_two[1].query)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return cosine_sim > 0.90  # REPETITION_COSINE_THRESHOLD
```

### Supervisor Context Management

The Supervisor never sees raw evidence — only compact summaries.

```python
def build_supervisor_context(state: MASISState) -> dict:
    return {
        "original_query": state["original_query"],
        "dag_overview": [
            {"id": t.task_id, "type": t.type, "status": t.status}
            for t in state["task_dag"]
        ],
        "last_result": {
            "task_id": state["last_task_result"].task_id,
            "summary": state["last_task_result"].summary[:500],  # truncated
            "criteria_result": state["last_task_result"].criteria_result,
        },
        "budget": {
            "tokens_used": state["token_budget"].total_tokens_used,
            "tokens_remaining": state["token_budget"].remaining,
            "cost_usd": state["token_budget"].total_cost_usd,
        },
        "iteration": state["iteration_count"],
    }
# This context is ~800 tokens vs 10,000+ if full evidence were included
```

---

## Executor Internals

### Per-Agent State Filtering

```python
def filtered_state(state: MASISState, task: TaskNode) -> dict:
    if task.type == "researcher":
        return {
            "original_query": state["original_query"],
            "task_query": task.query,
            # Does NOT get: evidence_board, critique_notes, other task results
        }
    elif task.type == "skeptic":
        return {
            "original_query": state["original_query"],
            "evidence_board": state["evidence_board"],  # sees ALL evidence by design
            "task_dag": state["task_dag"],
        }
    elif task.type == "synthesizer":
        return {
            "original_query": state["original_query"],
            "evidence_board": state["evidence_board"],
            "critique_notes": state.get("critique_notes", []),
            "task_dag": state["task_dag"],
        }
    elif task.type == "web_search":
        return {"task_query": task.query}   # minimal
```

### Timeout and Error Handling

```python
AGENT_TIMEOUTS = {
    "researcher": 30, "web_search": 15, "skeptic": 45, "synthesizer": 60,
}

async def dispatch_with_safety(task: TaskNode, state: MASISState) -> AgentOutput:
    timeout = AGENT_TIMEOUTS.get(task.type, 30)
    try:
        result = await asyncio.wait_for(
            dispatch_agent(task, filtered_state(state, task)),
            timeout=timeout
        )
        state["token_budget"].total_tokens_used += result.tokens_used
        state["token_budget"].api_calls[task.type] = \
            state["token_budget"].api_calls.get(task.type, 0) + 1
        task.status = "done"
        return result

    except asyncio.TimeoutError:
        task.status = "failed"
        return AgentOutput(
            task_id=task.task_id, task_type=task.type, status="failed",
            summary=f"Timeout after {timeout}s",
            criteria_result={"verdict": "FAIL", "reason": "timeout"},
        )
    except Exception as e:
        task.status = "failed"
        return AgentOutput(
            task_id=task.task_id, task_type=task.type, status="failed",
            summary=f"Error: {str(e)[:200]}",
            criteria_result={"verdict": "FAIL", "reason": str(type(e).__name__)},
        )
```

---

## Persistence — PostgresSaver

For HITL resume and fault tolerance, use PostgresSaver instead of InMemorySaver.

```bash
# Local Docker (dev)
docker run -d --name masis-postgres \
  -e POSTGRES_USER=masis \
  -e POSTGRES_PASSWORD=masis_dev \
  -e POSTGRES_DB=masis_checkpoints \
  -p 5432:5432 postgres:16

# .env
POSTGRES_URL=postgresql://masis:masis_dev@localhost:5432/masis_checkpoints
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.getenv("POSTGRES_URL")

# Sync (scripts/tests):
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    result = graph.invoke({"original_query": "Q3 revenue?"}, config)

# Async (FastAPI):
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    result = await graph.ainvoke({"original_query": "Q3 revenue?"}, config)
```

What checkpointing enables:
- **HITL resume**: `interrupt()` saves state. User responds hours later. `Command(resume=...)` continues exactly where it paused.
- **Fault tolerance**: Process crash → restart with same `thread_id` → resumes from last checkpoint.
- **Time travel**: `graph.get_state_history(config)` → see every state → fork from any point.
- **Audit trail**: Every decision, evidence chunk, and quality score persisted.

---

## Relevant Code

| Component | File |
|---|---|
| State schema | `masis/schemas/models.py` |
| Graph wiring | `masis/graph/workflow.py` |
| Routing edges | `masis/graph/edges.py` |
| Supervisor modes | `masis/nodes/supervisor.py` |
| Evidence reducer | `masis/schemas/models.py` — `evidence_reducer()` |
