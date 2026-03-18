"""
Main Pipeline Orchestrator for Multi-Agent Workflows.

Connects RAG Agent → Content Agent into a production-ready pipeline.
Designed for:
- Standalone CLI usage
- FastAPI backend integration
- Future LangGraph migration
- Email Agent integration

Core Principles:
- Each stage behaves like a node (LangGraph-ready)
- Inputs/outputs are clean and structured
- Agents remain independently usable
- All configuration is externalized
- Comprehensive logging and error handling
- Timing metrics for performance tracking
"""

import time
import json
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from agents.rag_agent import rag_agent
from agents.content_agent import content_agent, _load_config as load_content_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PipelineMetrics:
    """Timing and performance metrics for pipeline execution."""
    
    total_duration: float = 0.0
    rag_duration: float = 0.0
    content_duration: float = 0.0
    validation_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class PipelineResponse:
    """Structured response from pipeline execution."""
    
    success: bool
    query: str
    rag_output: Optional[str] = None
    final_output: Optional[str] = None
    persona: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PipelineError(Exception):
    """Base exception for pipeline execution errors."""
    pass


class ValidationError(PipelineError):
    """Raised when input validation fails."""
    pass


class RAGStageError(PipelineError):
    """Raised when RAG stage fails."""
    pass


class ContentStageError(PipelineError):
    """Raised when Content stage fails."""
    pass


def _load_content_config_safe() -> Dict[str, Any]:
    """
    Load content agent config safely.
    
    Returns:
        Config dictionary
        
    Raises:
        ValidationError: If config loading fails (but not agent failure)
    """
    try:
        return load_content_config()
    except Exception as e:
        logger.error(f"Failed to load content agent config: {str(e)}")
        raise ValidationError(f"Configuration error: {str(e)}") from e


def validate_inputs(
    query: str,
    content_type: str,
    persona: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate pipeline inputs against configuration.
    
    Args:
        query: User query/input text
        content_type: Type of content to generate
        persona: Target persona for content
        config: Content agent config (loaded if not provided)
        
    Raises:
        ValidationError: If any validation fails
    """
    # Validate query
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty.")
    
    if len(query) > 50000:
        raise ValidationError("Query exceeds maximum length (50000 characters).")
    
    # Load config if not provided
    if config is None:
        config = _load_content_config_safe()
    
    # Validate persona
    available_personas = config.get("personas", {})
    if persona not in available_personas:
        raise ValidationError(
            f"Invalid persona '{persona}'. Available: {', '.join(available_personas.keys())}"
        )
    
    # Validate content type
    available_types = config.get("content_types", {})
    if content_type not in available_types:
        raise ValidationError(
            f"Invalid content_type '{content_type}'. Available: {', '.join(available_types.keys())}"
        )


def _rag_stage(query: str, use_rag: bool, debug: bool = False) -> tuple[str, float]:
    """
    Execute RAG retrieval stage.
    
    Args:
        query: User query
        use_rag: Whether to execute RAG or skip to content stage
        debug: Enable debug logging
        
    Returns:
        Tuple of (rag_output, duration_seconds)
        
    Raises:
        RAGStageError: If RAG execution fails
    """
    start_time = time.time()
    
    if not use_rag:
        if debug:
            logger.debug("[RAG STAGE] Skipped (use_rag=False)")
        return query, time.time() - start_time
    
    try:
        if debug:
            logger.debug(f"[RAG STAGE] Starting RAG retrieval for query: {query[:100]}")
        
        logger.info("[PIPELINE] Running RAG stage")
        rag_output = rag_agent(query)
        
        if debug:
            logger.debug(f"[RAG STAGE] Completed. Output length: {len(rag_output)} chars")
        
        duration = time.time() - start_time
        logger.info(f"[PIPELINE] RAG stage completed in {duration:.2f}s")
        
        return rag_output, duration
        
    except Exception as e:
        logger.error(f"[PIPELINE] RAG stage failed: {str(e)}")
        raise RAGStageError(f"RAG retrieval failed: {str(e)}") from e


def _content_stage(
    rag_output: str,
    content_type: str,
    persona: str,
    debug: bool = False
) -> tuple[str, float]:
    """
    Execute content transformation stage.
    
    Args:
        rag_output: Output from RAG stage
        content_type: Type of content to generate
        persona: Target persona
        debug: Enable debug logging
        
    Returns:
        Tuple of (content_output, duration_seconds)
        
    Raises:
        ContentStageError: If content generation fails
    """
    start_time = time.time()
    
    try:
        if debug:
            logger.debug(
                f"[CONTENT STAGE] Starting content generation. "
                f"Type={content_type}, Persona={persona}, Input length={len(rag_output)}"
            )
        
        logger.info("[PIPELINE] Running Content stage")
        content_output = content_agent(
            input_text=rag_output,
            content_type=content_type,
            persona=persona
        )
        
        if debug:
            logger.debug(f"[CONTENT STAGE] Completed. Output length: {len(content_output)} chars")
        
        duration = time.time() - start_time
        logger.info(f"[PIPELINE] Content stage completed in {duration:.2f}s")
        
        return content_output, duration
        
    except Exception as e:
        logger.error(f"[PIPELINE] Content stage failed: {str(e)}")
        raise ContentStageError(f"Content generation failed: {str(e)}") from e


def run_pipeline(
    query: str,
    content_type: str = "summary",
    persona: str = "technical_writer",
    use_rag: bool = True,
    debug: bool = False
) -> PipelineResponse:
    """
    Execute full pipeline: Validate → RAG → Content → Return.
    
    Production-ready main pipeline function. Agents remain independently usable.
    Designed for FastAPI integration and future LangGraph migration.
    
    Args:
        query: User query or factual input (required)
        content_type: Type of content to generate (default: "summary")
                     Must be in content_agent_config.json
        persona: Target persona for content (default: "technical_writer")
                Must be in content_agent_config.json
        use_rag: Whether to run RAG stage (default: True)
                If False, query is passed directly to content stage
        debug: Enable debug-level logging (default: False)
    
    Returns:
        PipelineResponse object containing:
        - success: Boolean indicating overall success
        - query: Original user query
        - rag_output: Output from RAG stage (None if use_rag=False)
        - final_output: Final structured content
        - persona: Persona used
        - content_type: Content type generated
        - error: Error message if failed
        - metrics: Timing metrics
        - timestamp: ISO format timestamp
    
    Examples:
        # With RAG (default):
        response = run_pipeline("Explain machine learning")
        
        # Without RAG (direct content transformation):
        response = run_pipeline(
            "RAG output facts: ML is...",
            use_rag=False,
            persona="beginner_teacher"
        )
        
        # With debug logging:
        response = run_pipeline("Query", debug=True)
    
    Note:
        - Each stage can be called independently if needed
        - All configuration is externalized to content_agent_config.json
        - Metrics are tracked for performance analysis
        - Comprehensive error handling prevents crashes
    """
    pipeline_start = time.time()
    metrics = PipelineMetrics()
    
    logger.info(
        "[PIPELINE] Starting execution | query_len=%s use_rag=%s content_type=%s persona=%s",
        len(query), use_rag, content_type, persona
    )
    
    if debug:
        logger.debug(f"[PIPELINE DEBUG] Query: {query[:200]}")
    
    try:
        # ============================================================
        # STAGE 1: VALIDATION
        # ============================================================
        validation_start = time.time()
        
        if debug:
            logger.debug("[PIPELINE] Starting input validation")
        
        config = _load_content_config_safe()
        validate_inputs(query, content_type, persona, config)
        
        metrics.validation_duration = time.time() - validation_start
        logger.info(f"[PIPELINE] Validation completed in {metrics.validation_duration:.2f}s")
        
        # ============================================================
        # STAGE 2: RAG RETRIEVAL (optional)
        # ============================================================
        rag_output, metrics.rag_duration = _rag_stage(query, use_rag, debug)
        
        # ============================================================
        # STAGE 3: CONTENT TRANSFORMATION
        # ============================================================
        final_output, metrics.content_duration = _content_stage(
            rag_output, content_type, persona, debug
        )
        
        # ============================================================
        # STAGE 4: RETURN STRUCTURED RESPONSE
        # ============================================================
        metrics.total_duration = time.time() - pipeline_start
        
        logger.info(
            "[PIPELINE] Execution completed successfully | "
            "total_duration=%.2fs validation=%.2fs rag=%.2fs content=%.2fs | "
            "query_len=%s output_len=%s",
            metrics.total_duration,
            metrics.validation_duration,
            metrics.rag_duration,
            metrics.content_duration,
            len(query),
            len(final_output)
        )
        
        return PipelineResponse(
            success=True,
            query=query,
            rag_output=rag_output if use_rag else None,
            final_output=final_output,
            persona=persona,
            content_type=content_type,
            error=None,
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except ValidationError as e:
        logger.warning(f"[PIPELINE] Validation error: {str(e)}")
        metrics.total_duration = time.time() - pipeline_start
        
        return PipelineResponse(
            success=False,
            query=query,
            error=str(e),
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except (RAGStageError, ContentStageError) as e:
        logger.error(f"[PIPELINE] Stage error: {str(e)}")
        metrics.total_duration = time.time() - pipeline_start
        
        return PipelineResponse(
            success=False,
            query=query,
            error=str(e),
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"[PIPELINE] Unexpected error: {str(e)}", exc_info=True)
        metrics.total_duration = time.time() - pipeline_start
        
        return PipelineResponse(
            success=False,
            query=query,
            error=f"Unexpected error: {str(e)}",
            metrics=metrics.to_dict(),
            timestamp=datetime.utcnow().isoformat()
        )


def get_available_options() -> Dict[str, list]:
    """
    Get available personas and content types from configuration.
    
    Returns:
        Dictionary with 'personas' and 'content_types' lists
    """
    try:
        config = _load_content_config_safe()
        return {
            "personas": list(config.get("personas", {}).keys()),
            "content_types": list(config.get("content_types", {}).keys())
        }
    except Exception as e:
        logger.error(f"Failed to get available options: {str(e)}")
        raise
