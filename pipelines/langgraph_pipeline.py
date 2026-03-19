"""
LangGraph-based pipeline orchestrator for multi-agent workflows.

This module introduces a graph orchestration layer while preserving:
- Independent agent usability (rag_agent, content_agent)
- Existing config-driven validation
- Existing CLI interfaces (opt-in via --use-langgraph)
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.content_agent import content_agent
from agents.email_agent import email_agent
from agents.rag_agent import rag_agent
from utils.logger import setup_logger
from pipelines.main_pipeline import (
    ContentStageError,
    EmailStageError,
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
    email: Optional[str]
    email_subject: Optional[str]
    email_status: Optional[Dict[str, Any]]
    _rag_duration: float
    _content_duration: float
    _email_duration: float


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
        "email",
        "email_subject",
        "email_status",
        "_rag_duration",
        "_content_duration",
        "_email_duration",
    ]

    missing = [field for field in required_fields if field not in state]
    if missing:
        raise GraphStateError(f"Missing state fields: {', '.join(missing)}")

    for field in ["query", "rag_output", "final_output", "persona", "content_type"]:
        if not isinstance(state[field], str):
            raise GraphStateError(f"Invalid type for '{field}': expected str")

    if not isinstance(state["use_rag"], bool):
        raise GraphStateError("Invalid type for 'use_rag': expected bool")

    if state["email"] is not None and not isinstance(state["email"], str):
        raise GraphStateError("Invalid type for 'email': expected Optional[str]")

    if state["email_subject"] is not None and not isinstance(state["email_subject"], str):
        raise GraphStateError("Invalid type for 'email_subject': expected Optional[str]")

    if state["email_status"] is not None and not isinstance(state["email_status"], dict):
        raise GraphStateError("Invalid type for 'email_status': expected Optional[dict]")

    for field in ["_rag_duration", "_content_duration", "_email_duration"]:
        if not isinstance(state[field], (int, float)):
            raise GraphStateError(f"Invalid type for '{field}': expected float")


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

        state["_rag_duration"] = time.time() - start
        logger.info(
            "[LANGGRAPH] Exit node: rag_node | duration=%.2fs output_len=%s | Transition: rag_node -> content_node",
            state["_rag_duration"],
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

        state["_content_duration"] = time.time() - start
        logger.info(
            "[LANGGRAPH] Exit node: content_node | duration=%.2fs output_len=%s | Transition: content_node -> email_node",
            state["_content_duration"],
            len(state["final_output"]),
        )
        return state

    except Exception as error:
        logger.error("[LANGGRAPH] content_node failed: %s", str(error))
        raise ContentStageError(f"Content node failed: {error}") from error


def email_node(state: PipelineState) -> PipelineState:
    """Email node. Sends final_output if email recipient is provided."""
    _validate_state(state)

    start = time.time()
    logger.info("[LANGGRAPH] Enter node: email_node")

    try:
        recipient = (state.get("email") or "").strip()
        if not recipient:
            state["email_status"] = {
                "status": "skipped",
                "recipient": "",
                "subject": "",
                "error": None,
            }
            state["_email_duration"] = time.time() - start
            logger.info(
                "[LANGGRAPH] Exit node: email_node | duration=%.2fs status=skipped | Transition: email_node -> END",
                state["_email_duration"],
            )
            return state

        status = email_agent(
            content=state["final_output"],
            recipient_email=recipient,
            subject=state.get("email_subject"),
        )
        state["email_status"] = status
        state["_email_duration"] = time.time() - start

        if status.get("status") != "success":
            raise EmailStageError(f"Email node failed: {status.get('error', 'unknown error')}")

        logger.info(
            "[LANGGRAPH] Exit node: email_node | duration=%.2fs status=success | Transition: email_node -> END",
            state["_email_duration"],
        )
        return state

    except EmailStageError:
        raise
    except Exception as error:
        logger.error("[LANGGRAPH] email_node failed: %s", str(error))
        raise EmailStageError(f"Email node failed: {error}") from error


def build_langgraph() -> Any:
    """
    Build and compile the LangGraph workflow.

    Current flow:
    rag_node -> content_node -> email_node -> END

    Future-ready extension points:
    - Add email node after content_node
    - Add conditional branching and retries
    - Add memory integration nodes
    """
    graph = StateGraph(PipelineState)

    graph.add_node("rag_node", rag_node)
    graph.add_node("content_node", content_node)
    graph.add_node("email_node", email_node)

    graph.set_entry_point("rag_node")
    graph.add_edge("rag_node", "content_node")
    graph.add_edge("content_node", "email_node")
    graph.add_edge("email_node", END)

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
    send_email: bool = False,
    email: Optional[str] = None,
    email_subject: Optional[str] = None,
) -> PipelineResponse:
    """
    Execute LangGraph-based pipeline.

    Returns a PipelineResponse compatible with the existing pipeline interface.
    """
    pipeline_start = time.time()
    metrics = PipelineMetrics()

    logger.info(
        "[LANGGRAPH] Pipeline start | query_len=%s use_rag=%s content_type=%s persona=%s send_email=%s",
        len(query),
        use_rag,
        content_type,
        persona,
        send_email or bool(email),
    )

    if debug:
        logger.debug("[LANGGRAPH DEBUG] Query preview: %s", query[:200])

    try:
        validation_start = time.time()
        config = _load_content_config_safe()
        validate_inputs(query, content_type, persona, config)

        send_email_requested = send_email or bool(email)
        if send_email_requested and not email:
            raise ValidationError("Email recipient is required when email delivery is enabled.")

        metrics.validation_duration = time.time() - validation_start

        initial_state: PipelineState = {
            "query": query,
            "rag_output": "",
            "final_output": "",
            "persona": persona,
            "content_type": content_type,
            "use_rag": use_rag,
            "email": email if send_email_requested else None,
            "email_subject": email_subject,
            "email_status": None,
            "_rag_duration": 0.0,
            "_content_duration": 0.0,
            "_email_duration": 0.0,
        }
        _validate_state(initial_state)

        logger.info("[LANGGRAPH] Transition: START -> rag_node")
        graph = _get_compiled_graph()
        result_state = graph.invoke(initial_state)
        _validate_state(result_state)

        metrics.rag_duration = float(result_state.get("_rag_duration", 0.0))
        metrics.content_duration = float(result_state.get("_content_duration", 0.0))
        metrics.email_duration = float(result_state.get("_email_duration", 0.0))
        metrics.total_duration = time.time() - pipeline_start

        logger.info(
            "[LANGGRAPH] Pipeline completed | total=%.2fs validation=%.2fs rag=%.2fs content=%.2fs email=%.2fs",
            metrics.total_duration,
            metrics.validation_duration,
            metrics.rag_duration,
            metrics.content_duration,
            metrics.email_duration,
        )

        return PipelineResponse(
            success=True,
            query=query,
            rag_output=result_state["rag_output"] if use_rag else None,
            final_output=result_state["final_output"],
            persona=persona,
            content_type=content_type,
            email_status=result_state.get("email_status"),
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

    except (GraphStateError, RAGStageError, ContentStageError, EmailStageError) as error:
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