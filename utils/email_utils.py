"""
Email utility helpers for SMTP delivery.

This module is intentionally isolated so SMTP can be swapped later
(OAuth/Gmail API, providers, templates, attachments, bulk delivery).
"""

import os
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

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

    message = MIMEMultipart("alternative")
    message["From"] = email_settings.user
    message["To"] = recipient_email
    message["Subject"] = subject

    message.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        message.attach(MIMEText(body_html, "html", "utf-8"))

    logger.info("[EMAIL] Preparing SMTP client host=%s port=%s", email_settings.host, email_settings.port)

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
