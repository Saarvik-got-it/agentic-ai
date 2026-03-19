"""
LangGraph-based pipeline orchestrator for multi-agent workflows.

This module introduces a graph orchestration layer while preserving:
- Independent agent usability (rag_agent, content_agent)
- Existing config-driven validation
- Existing CLI interfaces (opt-in via --use-langgraph)
"""

import time
from datetime import datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.content_agent import content_agent
from agents.rag_agent import rag_agent
from utils.logger import setup_logger
from pipelines.main_pipeline import (
    ContentStageError,
    PipelineMetrics,
    PipelineResponse,
    RAGStageError,
    ValidationError,
    _load_content_config_safe,
    validate_inputs,
)

logger = setup_logger(__name__)


class PipelineState(TypedDict):
    """Shared state passed across LangGraph pipeline nodes."""

    query: str
    rag_output: str
    final_output: str
    persona: str
    content_type: str
    use_rag: bool


class GraphStateError(Exception):
    """Raised when pipeline graph state is invalid."""


_compiled_graph: Optional[Any] = None


def _validate_state(state: PipelineState) -> None:
    """Validate required state fields and types."""
    required_fields = [
        "query",
        "rag_output",
        "final_output",
        "persona",
        "content_type",
        "use_rag",
    ]

    missing = [field for field in required_fields if field not in state]
    if missing:
        raise GraphStateError(f"Missing state fields: {', '.join(missing)}")

    for field in ["query", "rag_output", "final_output", "persona", "content_type"]:
        if not isinstance(state[field], str):
            raise GraphStateError(f"Invalid type for '{field}': expected str")

    if not isinstance(state["use_rag"], bool):
        raise GraphStateError("Invalid type for 'use_rag': expected bool")


def rag_node(state: PipelineState) -> PipelineState:
    """
    RAG node.

    - If use_rag=True: calls rag_agent(query)
    - If use_rag=False: forwards query directly
    """
    _validate_state(state)

    start = time.time()
    logger.info("[LANGGRAPH] Enter node: rag_node")

    try:
        if state["use_rag"]:
            logger.info("[LANGGRAPH] rag_node executing RAG retrieval")
            state["rag_output"] = rag_agent(state["query"])
        else:
            logger.info("[LANGGRAPH] rag_node bypassing retrieval (use_rag=False)")
            state["rag_output"] = state["query"]

        state["_rag_duration"] = time.time() - start  # type: ignore[index]
        logger.info(
            "[LANGGRAPH] Exit node: rag_node | duration=%.2fs output_len=%s | Transition: rag_node -> content_node",
            state["_rag_duration"],  # type: ignore[index]
            len(state["rag_output"]),
        )
        return state

    except Exception as error:
        logger.error("[LANGGRAPH] rag_node failed: %s", str(error))
        raise RAGStageError(f"RAG node failed: {error}") from error


def content_node(state: PipelineState) -> PipelineState:
    """Content node. Transforms rag_output into final_output."""
    _validate_state(state)

    start = time.time()
    logger.info("[LANGGRAPH] Enter node: content_node")

    try:
        state["final_output"] = content_agent(
            input_text=state["rag_output"],
            content_type=state["content_type"],
            persona=state["persona"],
        )

        state["_content_duration"] = time.time() - start  # type: ignore[index]
        logger.info(
            "[LANGGRAPH] Exit node: content_node | duration=%.2fs output_len=%s | Transition: content_node -> END",
            state["_content_duration"],  # type: ignore[index]
            len(state["final_output"]),
        )
        return state

    except Exception as error:
        logger.error("[LANGGRAPH] content_node failed: %s", str(error))
        raise ContentStageError(f"Content node failed: {error}") from error


def build_langgraph() -> Any:
    """
    Build and compile the LangGraph workflow.

    Current flow:
      rag_node -> content_node -> END

    Future-ready extension points:
    - Add email node after content_node
    - Add conditional branching and retries
    - Add memory integration nodes
    """
    graph = StateGraph(PipelineState)

    graph.add_node("rag_node", rag_node)
    graph.add_node("content_node", content_node)

    graph.set_entry_point("rag_node")
    graph.add_edge("rag_node", "content_node")
    graph.add_edge("content_node", END)

    return graph.compile()


def _get_compiled_graph() -> Any:
    """Lazily initialize and cache compiled graph for reuse."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("[LANGGRAPH] Compiling graph")
        _compiled_graph = build_langgraph()
    return _compiled_graph


def run_langgraph_pipeline(
    query: str,
    content_type: str = "summary",
    persona: str = "technical_writer",
    use_rag: bool = True,
    debug: bool = False,
) -> PipelineResponse:
    """
    Execute LangGraph-based pipeline.

    Returns a PipelineResponse compatible with the existing pipeline interface.
    """
    pipeline_start = time.time()
    metrics = PipelineMetrics()

    logger.info(
        "[LANGGRAPH] Pipeline start | query_len=%s use_rag=%s content_type=%s persona=%s",
        len(query),
        use_rag,
        content_type,
        persona,
    )

    if debug:
        logger.debug("[LANGGRAPH DEBUG] Query preview: %s", query[:200])

    try:
        validation_start = time.time()
        config = _load_content_config_safe()
        validate_inputs(query, content_type, persona, config)
        metrics.validation_duration = time.time() - validation_start

        initial_state: PipelineState = {
            "query": query,
            "rag_output": "",
            "final_output": "",
            "persona": persona,
            "content_type": content_type,
            "use_rag": use_rag,
        }
        _validate_state(initial_state)

        logger.info("[LANGGRAPH] Transition: START -> rag_node")
        graph = _get_compiled_graph()
        result_state = graph.invoke(initial_state)
        _validate_state(result_state)

        metrics.rag_duration = float(result_state.get("_rag_duration", 0.0))
        metrics.content_duration = float(result_state.get("_content_duration", 0.0))
        metrics.total_duration = time.time() - pipeline_start

        logger.info(
            "[LANGGRAPH] Pipeline completed | total=%.2fs validation=%.2fs rag=%.2fs content=%.2fs",
            metrics.total_duration,
            metrics.validation_duration,
            metrics.rag_duration,
            metrics.content_duration,
        )

        return PipelineResponse(
            success=True,
            query=query,
            rag_output=result_state["rag_output"] if use_rag else None,
            final_output=result_state["final_output"],
            persona=persona,
            content_type=content_type,
            error=None,
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat(),
        )

    except ValidationError as error:
        metrics.total_duration = time.time() - pipeline_start
        logger.warning("[LANGGRAPH] Validation error: %s", str(error))
        return PipelineResponse(
            success=False,
            query=query,
            error=str(error),
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat(),
        )

    except (GraphStateError, RAGStageError, ContentStageError) as error:
        metrics.total_duration = time.time() - pipeline_start
        logger.error("[LANGGRAPH] Graph execution error: %s", str(error))
        return PipelineResponse(
            success=False,
            query=query,
            error=str(error),
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as error:
        metrics.total_duration = time.time() - pipeline_start
        logger.error("[LANGGRAPH] Unexpected error: %s", str(error), exc_info=True)
        return PipelineResponse(
            success=False,
            query=query,
            error=f"Unexpected error: {error}",
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat(),
        )