# MASIS Architecture Diagram

3-node LangGraph StateGraph: **Supervisor → Executor → Validator**. Agents (Researcher, Skeptic, Synthesizer) are Python functions dispatched by the Executor, not separate graph nodes.

```mermaid
flowchart TD
    U([User Query]) --> SUP

    subgraph SUP[Supervisor]
        FAST["Fast Path  —  budget · iteration · repetition · criteria check\n$0 · <10ms · runs every turn"]
        SLOW["Slow Path  —  retry · DAG modify · HITL · force synthesize\ngpt-4.1 · ~$0.015 · only on task failure"]
    end

    SUP -->|"dispatch tasks"| EXE

    subgraph EXE[Executor]
        R1[Researcher 1\nHyDE · CRAG · Self-RAG\nBM25 + vector]
        R2[Researcher 2\nWeb Search\nTavily]
        SK[Skeptic\nNLI + o3-mini judge]
        SY[Synthesizer\nU-shape · citations]

        R1 --> SK
        R2 --> SK
        SK --> SY
    end

    EXE -->|"criteria + AgentOutput per task"| SUP

    SY --> VAL

    subgraph VAL[Validator]
        V1["faithfulness · citation accuracy · relevancy · completeness\nAll pass → END    Any fail → back to Supervisor (max 2 rounds)"]
    end

    VAL -->|"revise"| SUP
    VAL -->|"pass"| ANS([Final Answer])
```

For full architecture details, see [docs_md/01_HLD.md](../../docs_md/01_HLD.md).
