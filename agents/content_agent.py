"""
Content Agent implementation module.
Transforms factual input into structured, persona-driven output for downstream channels.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils.llm import generate_with_fallback
from utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_CONFIG_PATH = "./config/content_agent_config.json"


class ContentAgentError(Exception):
    """Domain-specific exception for content agent failures."""


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load content agent configuration from JSON."""
    resolved_path = config_path or os.getenv("CONTENT_AGENT_CONFIG", DEFAULT_CONFIG_PATH)
    path = Path(resolved_path)

    if not path.exists():
        raise ContentAgentError(
            f"Config file not found at {resolved_path}. "
            "Set CONTENT_AGENT_CONFIG or create config/content_agent_config.json."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ContentAgentError(f"Invalid JSON in config file: {error}") from error
    except Exception as error:
        raise ContentAgentError(f"Unable to read config file: {error}") from error

    required_keys = ["personas", "content_types", "prompt"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ContentAgentError(f"Config missing required keys: {missing}")

    return config


def _validate_inputs(input_text: str, content_type: str, persona: str, config: Dict[str, Any]) -> None:
    """Validate runtime input and config-driven options."""
    if not input_text or not input_text.strip():
        raise ContentAgentError("Input text cannot be empty.")

    personas = config.get("personas", {})
    content_types = config.get("content_types", {})

    if persona not in personas:
        raise ContentAgentError(
            f"Invalid persona '{persona}'. Available personas: {', '.join(personas.keys())}"
        )

    if content_type not in content_types:
        raise ContentAgentError(
            f"Invalid content_type '{content_type}'. Available content types: {', '.join(content_types.keys())}"
        )


def build_prompt(input_text: str, content_type: str, persona: str) -> str:
    """
    Dynamically build a content prompt based on config-defined personas and structures.
    """
    config = _load_config()
    _validate_inputs(input_text, content_type, persona, config)

    persona_cfg = config["personas"][persona]
    content_cfg = config["content_types"][content_type]
    prompt_cfg = config["prompt"]

    template = prompt_cfg.get(
        "template",
        (
            "You are a content transformation agent.\n\n"
            "{system_instructions}\n\n"
            "Persona:\n"
            "- Name: {persona_name}\n"
            "- Tone: {persona_tone}\n"
            "- Depth: {persona_depth}\n"
            "- Formatting style: {persona_formatting_style}\n\n"
            "Requested content type:\n"
            "- Type: {content_type}\n"
            "- Objective: {content_objective}\n"
            "{structure_header}\n"
            "{content_structure}\n\n"
            "Rules:\n"
            "- {no_hallucination_rule}\n"
            "- {readability_rule}\n\n"
            "Input facts:\n"
            "{input_text}\n\n"
            "Produce the final content now."
        ),
    )

    return template.format(
        system_instructions=prompt_cfg.get("system_instructions", ""),
        persona_name=persona,
        persona_tone=persona_cfg.get("tone", "neutral"),
        persona_depth=persona_cfg.get("depth", "balanced"),
        persona_formatting_style=persona_cfg.get("formatting_style", "clear"),
        content_type=content_type,
        content_objective=content_cfg.get("objective", "Transform input into readable content."),
        structure_header=prompt_cfg.get("structure_header", "Required output structure:"),
        content_structure="\n".join(content_cfg.get("structure", [])),
        no_hallucination_rule=prompt_cfg.get("no_hallucination_rule", "Do not invent facts."),
        readability_rule=prompt_cfg.get("readability_rule", "Prioritize clarity and structure."),
        input_text=input_text.strip(),
    )


def _post_process_output(output_text: str, content_type: str, config: Dict[str, Any]) -> str:
    """Apply lightweight formatting and optional length control."""
    cleaned = (output_text or "").strip()

    if not cleaned:
        return ""

    max_chars = config.get("length_control", {}).get("max_chars_by_type", {}).get(content_type)
    if isinstance(max_chars, int) and max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "\n\n[Truncated due to configured length control]"

    return cleaned


def content_agent(input_text: str, content_type: str, persona: str) -> str:
    """
    Pipeline-ready core function.

    Example:
        result = content_agent(rag_output, "blog", "technical_writer")
    """
    try:
        config = _load_config()
        _validate_inputs(input_text, content_type, persona, config)

        logger.info(
            "Content agent start | persona=%s content_type=%s input_chars=%s",
            persona,
            content_type,
            len(input_text),
        )

        prompt = build_prompt(input_text=input_text, content_type=content_type, persona=persona)
        response = generate_with_fallback(prompt)

        if not response or not response.strip():
            raise ContentAgentError("LLM returned empty content.")

        output = _post_process_output(response, content_type, config)

        logger.info(
            "Content agent complete | persona=%s content_type=%s output_chars=%s",
            persona,
            content_type,
            len(output),
        )

        return output

    except ContentAgentError as error:
        logger.error("Content agent validation error: %s", str(error))
        return f"Content Agent Error: {error}"
    except Exception as error:
        logger.error("Content agent runtime error: %s", str(error))
        return f"Content Agent Error: Unable to process content. Details: {error}"


def _read_input_text(input_text: Optional[str], input_file: Optional[str]) -> str:
    """Read input from direct text or a file path."""
    if input_text and input_text.strip():
        return input_text.strip()

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise ContentAgentError(f"Input file not found: {input_file}")
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as error:
            raise ContentAgentError(f"Unable to read input file: {error}") from error

    raise ContentAgentError("Provide either --input-text or --input-file.")


def _list_available_options(config: Dict[str, Any]) -> str:
    """Render available personas and content types for CLI users."""
    personas = ", ".join(config.get("personas", {}).keys())
    content_types = ", ".join(config.get("content_types", {}).keys())
    return f"Available personas: {personas}\nAvailable content types: {content_types}"


def main_cli() -> int:
    """Standalone CLI entrypoint for the content agent."""
    parser = argparse.ArgumentParser(description="Standalone Content Agent")
    parser.add_argument("--input-text", type=str, help="Direct factual input text")
    parser.add_argument("--input-file", type=str, help="Path to file containing factual input")
    parser.add_argument("--content-type", type=str, help="Configured content type")
    parser.add_argument("--persona", type=str, help="Configured persona")
    parser.add_argument("--config", type=str, help="Optional config file path")
    parser.add_argument("--list-options", action="store_true", help="List available personas and content types")

    args = parser.parse_args()

    try:
        config = _load_config(args.config)

        if args.list_options:
            print(_list_available_options(config))
            return 0

        if not args.content_type or not args.persona:
            raise ContentAgentError(
                "Both --content-type and --persona are required unless --list-options is used."
            )

        input_data = _read_input_text(args.input_text, args.input_file)
        result = content_agent(input_data, args.content_type, args.persona)

        print("\n" + "=" * 80)
        print("Content Agent Output")
        print("=" * 80)
        print(result)
        print("=" * 80)
        return 0

    except ContentAgentError as error:
        print(f"Error: {error}")
        return 1
    except Exception as error:
        print(f"Fatal error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
