"""
Email Agent implementation.

Transforms generated content into a professional email format and sends it via SMTP.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from utils.email_utils import (
    EmailConfigError,
    EmailSendError,
    send_smtp_email,
    markdown_to_html,
    _build_html_email,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

_EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _is_valid_email(email: str) -> bool:
    """Return True when recipient email has a valid basic format."""
    if not email:
        return False
    return bool(_EMAIL_PATTERN.match(email.strip()))


def _generate_subject(content: str) -> str:
    """Generate a practical default subject from content when not provided."""
    first_line = ""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return "AI Generated Report"

    if len(first_line) > 70:
        first_line = first_line[:67].rstrip() + "..."

    return f"AI Generated: {first_line}"


def _format_email_body(content: str) -> str:
    """Format content into a clean professional email body (plain text)."""
    clean_content = content.strip()
    return (
        "Hello,\n\n"
        "Here is your requested content:\n\n"
        f"{clean_content}\n\n"
        "Best regards,\n"
        "AI Assistant"
    )


def _format_html_email_body(content: str) -> str:
    """Format content into a professional HTML email with Markdown conversion.
    
    Converts Markdown to HTML and wraps in professional email template.
    """
    clean_content = content.strip()
    # Convert Markdown to HTML
    html_content = markdown_to_html(clean_content)
    lower_html = html_content.lower()
    logger.info(
        "[EMAIL] HTML conversion check h1=%s ul=%s strong=%s",
        "<h1" in lower_html,
        "<ul" in lower_html,
        "<strong" in lower_html,
    )
    logger.debug("[EMAIL] Converted HTML snippet: %s", html_content[:240].replace("\n", " "))
    # Wrap in professional email template
    return _build_html_email(html_content)


def email_agent(
    content: str,
    recipient_email: str,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send generated content via email.

    Returns a structured response:
      {
        "status": "success" | "failure",
        "recipient": "...",
        "subject": "...",
        "error": None | "message"
      }
    """
    try:
        if not content or not content.strip():
            raise ValueError("content must not be empty")

        if not _is_valid_email(recipient_email):
            raise ValueError(f"Invalid recipient_email format: {recipient_email}")

        resolved_subject = (subject or "").strip() or _generate_subject(content)
        body_text = _format_email_body(content)
        body_html = _format_html_email_body(content)

        logger.info("[EMAIL] Preparing email")
        logger.info("[EMAIL] Sending to %s", recipient_email)

        send_smtp_email(
            recipient_email=recipient_email.strip(),
            subject=resolved_subject,
            body_text=body_text,
            body_html=body_html,
        )

        logger.info("[EMAIL] Success")
        return {
            "status": "success",
            "recipient": recipient_email.strip(),
            "subject": resolved_subject,
            "error": None,
        }

    except (ValueError, EmailConfigError, EmailSendError) as error:
        logger.error("[EMAIL] Failure: %s", str(error))
        return {
            "status": "failure",
            "recipient": (recipient_email or "").strip(),
            "subject": (subject or "").strip() or "AI Generated Report",
            "error": str(error),
        }
    except Exception as error:
        logger.error("[EMAIL] Failure: %s", str(error), exc_info=True)
        return {
            "status": "failure",
            "recipient": (recipient_email or "").strip(),
            "subject": (subject or "").strip() or "AI Generated Report",
            "error": f"Unexpected error: {error}",
        }


def _read_input_text(input_text: Optional[str], input_file: Optional[str]) -> str:
    """Read email content from direct text or file input."""
    if input_text and input_text.strip():
        return input_text.strip()

    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise ValueError(f"Input file not found: {input_file}")
        return path.read_text(encoding="utf-8").strip()

    raise ValueError("Provide either --input-text or --input-file")


def main_cli() -> int:
    """Standalone CLI entrypoint for the Email Agent."""
    parser = argparse.ArgumentParser(description="Standalone Email Agent")
    parser.add_argument("--input-text", type=str, help="Direct content text")
    parser.add_argument("--input-file", type=str, help="Path to file containing content")
    parser.add_argument("--to", type=str, required=True, help="Recipient email address")
    parser.add_argument("--subject", type=str, help="Optional subject line")

    args = parser.parse_args()

    try:
        content = _read_input_text(args.input_text, args.input_file)
        result = email_agent(content=content, recipient_email=args.to, subject=args.subject)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "success" else 1
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failure",
                    "recipient": (args.to or "").strip(),
                    "subject": (args.subject or "").strip() or "AI Generated Report",
                    "error": str(error),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
