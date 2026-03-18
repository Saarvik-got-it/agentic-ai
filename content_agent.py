"""Compatibility wrapper for the Content Agent.

Implementation now lives under agents/content_agent.py, while this file keeps:
- python content_agent.py CLI usage
- from content_agent import content_agent imports
"""

from agents.content_agent import (  # noqa: F401
    ContentAgentError,
    build_prompt,
    content_agent,
    main_cli,
)


if __name__ == "__main__":
    raise SystemExit(main_cli())
