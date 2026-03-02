"""
masis.utils
===========
Utility functions used across the MASIS system.

This package provides three modules:

dag_utils
---------
Functions for walking and querying the dynamic task DAG:
    get_next_ready_tasks()  -- find tasks ready to execute (MF-SUP-08)
    all_tasks_done()        -- check if the DAG is fully complete
    find_task()             -- look up a TaskNode by task_id
    check_agent_criteria()  -- Fast Path criteria checker (MF-SUP-07)

text_utils
----------
Text manipulation utilities:
    u_shape_order()   -- reorder evidence for lost-in-the-middle mitigation (MF-SYN-01)
    is_repetitive()   -- cosine-similarity-based repetition detection (MF-SUP-06)

safety_utils
------------
Safety and audit utilities:
    content_sanitizer()  -- strip prompt injection patterns (MF-SAFE-04)
    log_decision()       -- append a structured entry to the decision log (MF-SUP-17)
"""

from masis.utils.dag_utils import (
    all_tasks_done,
    check_agent_criteria,
    find_task,
    get_next_ready_tasks,
)
from masis.utils.text_utils import is_repetitive, u_shape_order
from masis.utils.safety_utils import content_sanitizer, log_decision

__all__ = [
    # dag_utils
    "all_tasks_done",
    "check_agent_criteria",
    "find_task",
    "get_next_ready_tasks",
    # text_utils
    "is_repetitive",
    "u_shape_order",
    # safety_utils
    "content_sanitizer",
    "log_decision",
]
