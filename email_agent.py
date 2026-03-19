"""Compatibility wrapper for the Email Agent."""

from agents.email_agent import email_agent, main_cli  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main_cli())
