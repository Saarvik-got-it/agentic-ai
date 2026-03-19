"""
Email utility helpers for SMTP delivery.

This module is intentionally isolated so SMTP can be swapped later
(OAuth/Gmail API, providers, templates, attachments, bulk delivery).
"""

import os
import re
import smtplib
from html import escape
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

from dotenv import load_dotenv

from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailConfigError(Exception):
    """Raised when required email configuration is missing or invalid."""


class EmailSendError(Exception):
    """Raised when SMTP sending fails."""


@dataclass
class EmailSettings:
    """Runtime SMTP settings loaded from environment variables."""

    host: str
    port: int
    user: str
    password: str
    use_tls: bool = True


def load_email_settings() -> EmailSettings:
    """Load SMTP settings from environment with Gmail-compatible defaults."""
    load_dotenv()

    host = os.getenv("EMAIL_HOST", "smtp.gmail.com").strip()

    port_raw = os.getenv("EMAIL_PORT", "587").strip()
    try:
        port = int(port_raw)
    except ValueError as error:
        raise EmailConfigError(f"Invalid EMAIL_PORT value: {port_raw}") from error

    user = os.getenv("EMAIL_USER", "").strip()
    password = os.getenv("EMAIL_PASSWORD", "").strip()

    if not user:
        raise EmailConfigError("Missing EMAIL_USER in environment configuration.")

    if not password:
        raise EmailConfigError("Missing EMAIL_PASSWORD in environment configuration.")

    use_tls_raw = os.getenv("EMAIL_USE_TLS", "true").strip().lower()
    use_tls = use_tls_raw not in {"0", "false", "no", "off"}

    return EmailSettings(
        host=host,
        port=port,
        user=user,
        password=password,
        use_tls=use_tls,
    )


def send_smtp_email(
    recipient_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    settings: Optional[EmailSettings] = None,
) -> None:
    """Send an email via SMTP with optional HTML alternative body."""
    email_settings = settings or load_email_settings()

    message = _build_multipart_alternative_message(
        sender=email_settings.user,
        recipient=recipient_email,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
    )

    logger.info("[EMAIL] Preparing SMTP client host=%s port=%s", email_settings.host, email_settings.port)
    logger.info("[EMAIL] MIME top-level Content-Type: %s", message.get_content_type())

    try:
        with smtplib.SMTP(email_settings.host, email_settings.port, timeout=30) as server:
            server.ehlo()
            if email_settings.use_tls:
                server.starttls()
                server.ehlo()

            server.login(email_settings.user, email_settings.password)
            server.sendmail(email_settings.user, recipient_email, message.as_string())

    except smtplib.SMTPAuthenticationError as error:
        raise EmailSendError("SMTP authentication failed. Check EMAIL_USER and EMAIL_PASSWORD.") from error
    except smtplib.SMTPConnectError as error:
        raise EmailSendError("SMTP connection failed. Check EMAIL_HOST and EMAIL_PORT.") from error
    except smtplib.SMTPException as error:
        raise EmailSendError(f"SMTP error while sending email: {error}") from error
    except OSError as error:
        raise EmailSendError(f"Network error while sending email: {error}") from error


def _build_multipart_alternative_message(
    sender: str,
    recipient: str,
    subject: str,
    body_text: str,
    body_html: Optional[str],
) -> MIMEMultipart:
    """Build a strict multipart/alternative message for broad client compatibility."""
    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message["MIME-Version"] = "1.0"

    safe_text = (body_text or "").strip() or "No content provided."
    normalized_html = _normalize_html_body(body_html, safe_text)

    # RFC-compliant order for multipart/alternative:
    # 1) plain text (fallback), 2) HTML (preferred rich content)
    plain_part = MIMEText(safe_text, "plain", "utf-8")
    html_part = MIMEText(normalized_html, "html", "utf-8")
    message.attach(plain_part)
    message.attach(html_part)

    payload = message.get_payload() or []
    part_types = [part.get_content_type() for part in payload]
    logger.info("[EMAIL] MIME part order: %s", " -> ".join(part_types))
    logger.debug("[EMAIL] HTML snippet before send: %s", normalized_html[:240].replace("\n", " "))

    return message


def _normalize_html_body(body_html: Optional[str], fallback_text: str) -> str:
    """Guarantee we always send a valid HTML document part."""
    html_candidate = (body_html or "").strip()

    if not html_candidate:
        logger.warning("[EMAIL] Empty HTML body detected; generating HTML fallback from plain text")
        html_candidate = f"<pre>{escape(fallback_text)}</pre>"

    has_html_shell = bool(re.search(r"<html[\\s>]", html_candidate, flags=re.IGNORECASE)) and bool(
        re.search(r"<body[\\s>]", html_candidate, flags=re.IGNORECASE)
    )
    if not has_html_shell:
        logger.warning("[EMAIL] HTML body missing <html>/<body>; wrapping content")
        html_candidate = f"<html><body>{html_candidate}</body></html>"

    # Basic delivery-layer debugging to catch raw markdown leakage.
    has_structural_tags = any(tag in html_candidate.lower() for tag in ("<h1", "<h2", "<h3", "<ul", "<ol", "<strong", "<p"))
    if not has_structural_tags:
        logger.warning("[EMAIL] HTML body has no structural tags; content may still be markdown-like")

    return html_candidate


def markdown_to_html(md_text: str) -> str:
    """Convert Markdown text to HTML with professional styling.
    
    Args:
        md_text: Markdown formatted text
        
    Returns:
        HTML formatted text suitable for email
    """
    if not HAS_MARKDOWN:
        logger.warning("[EMAIL] Markdown package unavailable; using fallback converter")
        return _fallback_markdown_to_html(md_text)
    
    try:
        logger.info("[EMAIL] Converting Markdown to HTML")
        html = markdown.markdown(md_text, extensions=['extra', 'codehilite', 'tables'])
        return html
    except Exception as e:
        logger.warning("[EMAIL] Markdown conversion failed, using fallback converter: %s", str(e))
        return _fallback_markdown_to_html(md_text)


def _fallback_markdown_to_html(md_text: str) -> str:
    """Best-effort Markdown conversion when python-markdown isn't available."""
    lines = (md_text or "").splitlines()
    output = []
    in_ul = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if in_ul:
                output.append("</ul>")
                in_ul = False
            continue

        if stripped.startswith("### "):
            if in_ul:
                output.append("</ul>")
                in_ul = False
            output.append(f"<h3>{_inline_markdown_to_html(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            if in_ul:
                output.append("</ul>")
                in_ul = False
            output.append(f"<h2>{_inline_markdown_to_html(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            if in_ul:
                output.append("</ul>")
                in_ul = False
            output.append(f"<h1>{_inline_markdown_to_html(stripped[2:])}</h1>")
            continue

        if re.match(r"^[-*+]\s+", stripped):
            if not in_ul:
                output.append("<ul>")
                in_ul = True
            item_text = re.sub(r"^[-*+]\s+", "", stripped)
            output.append(f"<li>{_inline_markdown_to_html(item_text)}</li>")
            continue

        if in_ul:
            output.append("</ul>")
            in_ul = False

        output.append(f"<p>{_inline_markdown_to_html(stripped)}</p>")

    if in_ul:
        output.append("</ul>")

    return "\n".join(output)


def _inline_markdown_to_html(text: str) -> str:
    """Convert a small subset of inline markdown syntax to HTML."""
    safe = escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<em>\1</em>", safe)
    return safe


def _build_html_email(html_content: str) -> str:
    """Build a professional HTML email template.
    
    Args:
        html_content: HTML formatted content (converted from Markdown)
        
    Returns:
        Complete HTML email document with styling
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.25em; }}
        p {{ margin: 0 0 16px 0; }}
        ul, ol {{ margin: 0 0 16px 0; padding-left: 2em; }}
        li {{ margin: 0 0 8px 0; }}
        code {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background-color: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 85%;
        }}
        pre {{
            background-color: #f6f8fa;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            margin: 0 0 16px 0;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        blockquote {{
            margin: 0 0 16px 0;
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
        }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{
            border-collapse: collapse;
            margin: 0 0 16px 0;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        .email-header {{
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .email-footer {{
            border-top: 1px solid #e1e4e8;
            padding-top: 16px;
            margin-top: 24px;
            color: #6a737d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="email-header">
        <p>Hello,</p>
        <p>Here is your requested content:</p>
    </div>
    
    <div class="email-content">
        {html_content}
    </div>
    
    <div class="email-footer">
        <p>Best regards,<br><strong>AI Assistant</strong></p>
    </div>
</body>
</html>"""
