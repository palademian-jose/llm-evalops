"""Notifiers: stdout, Slack incoming webhook, SMTP email."""

from __future__ import annotations

import logging
import os
import smtplib
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

import httpx

from llmops.alerts.rules import CheckResult, NotifierConfig
from llmops.settings import Settings, get_settings

log = logging.getLogger("llmops.alerts")


def format_text(result: CheckResult, title: str = "llmops alert check") -> str:
    lines = [f"{title}: {len(result.violations)} violation(s)"]
    for v in result.violations:
        lines.append(
            f"- [{v.alert_id}] {v.rule} ({v.type}): baseline {v.baseline_value:.4g} -> "
            f"window {v.window_value:.4g}; {v.detail}; threshold {v.threshold}"
        )
    return "\n".join(lines)


class StdoutNotifier:
    def send(self, result: CheckResult) -> None:
        print(format_text(result))


class SlackNotifier:
    def __init__(self, webhook_url: str, post: Callable[..., Any] | None = None):
        self.webhook_url = webhook_url
        self._post = post or (lambda url, json: httpx.post(url, json=json, timeout=10.0))

    def send(self, result: CheckResult) -> None:
        resp = self._post(self.webhook_url, json={"text": format_text(result)})
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()


class EmailNotifier:
    def __init__(self, settings: Settings, to: list[str] | None = None, subject: str = ""):
        self.s = settings
        self.to = to or [a.strip() for a in (settings.alert_email_to or "").split(",") if a.strip()]
        self.subject = subject or "[llmops] alert rule violations"

    def send(self, result: CheckResult, smtp_factory: Callable[..., Any] | None = None) -> None:
        if not self.s.smtp_host or not self.to:
            raise RuntimeError("email notifier needs SMTP_HOST and ALERT_EMAIL_TO")
        msg = EmailMessage()
        msg["Subject"] = self.subject
        msg["From"] = self.s.alert_email_from or self.s.smtp_user or "llmops@localhost"
        msg["To"] = ", ".join(self.to)
        msg.set_content(format_text(result) + "\n\n" + result.to_markdown())
        factory = smtp_factory or smtplib.SMTP
        with factory(self.s.smtp_host, self.s.smtp_port, timeout=15) as smtp:
            if hasattr(smtp, "starttls"):
                try:
                    smtp.starttls()
                except smtplib.SMTPException:
                    pass
            if self.s.smtp_user and self.s.smtp_password:
                smtp.login(self.s.smtp_user, self.s.smtp_password)
            smtp.send_message(msg)


def build_notifiers(configs: list[NotifierConfig], settings: Settings | None = None) -> list[Any]:
    s = settings or get_settings()
    out: list[Any] = []
    for c in configs:
        if c.type == "stdout":
            out.append(StdoutNotifier())
        elif c.type == "slack":
            url = c.webhook_url or os.environ.get(c.webhook_url_env) or s.slack_webhook_url
            if not url:
                log.warning("slack notifier configured but no webhook URL set; skipping")
                continue
            out.append(SlackNotifier(url))
        elif c.type == "email":
            out.append(EmailNotifier(s, to=c.to, subject=c.subject))
    return out


def notify(
    result: CheckResult, configs: list[NotifierConfig], settings: Settings | None = None
) -> int:
    """Send the result through every configured notifier; returns the number that succeeded."""
    sent = 0
    for n in build_notifiers(configs, settings):
        try:
            n.send(result)
            sent += 1
        except Exception as e:  # noqa: BLE001 - one broken channel must not stop the others
            log.error("notifier %s failed: %s", type(n).__name__, e)
    return sent
