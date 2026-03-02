"""
masis.tests
===========
Phase 0 + Phase 1 test suite for the MASIS system.

Phase 0 test modules (foundation):
    test_models.py        -- Pydantic model validation, evidence_reducer, BudgetTracker
    test_thresholds.py    -- Threshold constant values and importability
    test_model_routing.py -- get_model(), get_fallback(), MODEL_ROUTING, TOOL_LIMITS
    test_settings.py      -- Settings loading, validation, env var handling
    test_dag_utils.py     -- get_next_ready_tasks(), check_agent_criteria(), all_tasks_done()
    test_text_utils.py    -- u_shape_order(), is_repetitive()
    test_safety_utils.py  -- content_sanitizer(), log_decision()

Phase 1 test modules (core components):
    test_nodes_supervisor  -- supervisor_node(), plan_dag(), monitor_and_route()
    test_nodes_executor    -- execute_dag_tasks(), dispatch_agent()
    test_nodes_validator   -- faithfulness, citation accuracy, relevancy, completeness
    test_agents_researcher -- hyde_rewrite(), hybrid retrieval, CRAG, Self-RAG
    test_agents_skeptic    -- extract_claims(), NLI pre-filter, LLM judge
    test_agents_synthesizer-- citation enforcement, SynthesizerOutput
    test_agents_web_search -- sanitize_content(), Tavily integration
"""
