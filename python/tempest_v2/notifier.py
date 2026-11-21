"""Tempest-NWP: Automated Notification Dispatcher for Forecast Runs.

Sends push notifications directly to the user's phone when:
- Observation ingestion completes
- Local observation-increment assimilation finishes
- Forecast integration completes
- Forecast encounters numerical instability or an error
- Google Cloud Storage upload completes

Supported Push Providers:
1. ntfy.sh (Recommended / Default Zero-Setup):
   - 100% Free, open source, no account or API keys required.
   - Install the ntfy app (Android/iOS), subscribe to your topic (e.g. 'tempest_nwp_jord').
   - Receive instant push notifications with sound, priority, and rich metrics.
   - Config: Env var TEMPEST_NOTIFY_TOPIC or CLI --notify-ntfy <topic>.
2. Telegram Bot:
   - Config: Env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
3. Webhook (Discord / Slack / Custom):
   - Config: Env var TEMPEST_WEBHOOK_URL.

Strictly 0% AI / 0% Machine Learning.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import urllib.error
import urllib.request

logger = logging.getLogger("tempest_v2.notifier")


def send_ntfy_notification(
    topic: str,
    title: str,
    message: str,
    priority: str = "default",
    tags: Optional[List[str]] = None,
    click_url: Optional[str] = None,
) -> bool:
    """Dispatches push notification via ntfy.sh."""
    url = f"https://ntfy.sh/{topic.strip()}"
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": priority,  # min, low, default, high, urgent
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url

    try:
        req = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            success = resp.status in (200, 201)
            if success:
                logger.info("Sent ntfy push notification to phone (topic: %s)", topic)
            return success
    except Exception as e:
        logger.warning("Failed to send ntfy push notification to %s: %s", topic, e)
        return False


def send_telegram_notification(
    bot_token: str,
    chat_id: str,
    text: str,
) -> bool:
    """Dispatches notification via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    payload = {
        "chat_id": chat_id.strip(),
        "text": text,
        "parse_mode": "Markdown",
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning("Failed to send Telegram notification: %s", e)
        return False


def send_webhook_notification(
    webhook_url: str,
    title: str,
    message: str,
) -> bool:
    """Dispatches notification via Discord/Slack webhook."""
    payload = {"content": f"**{title}**\n{message}"}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "Tempest-NWP/1.0"}
    try:
        req = urllib.request.Request(webhook_url.strip(), data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        logger.warning("Failed to dispatch webhook: %s", e)
        return False


def notify_run_event(
    event_type: str,  # 'start', 'success', 'warning', 'error', 'gcs_upload'
    run_id: str,
    forecast_hours: float,
    runtime_sec: float,
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    gcs_uri: Optional[str] = None,
    ntfy_topic: Optional[str] = None,
) -> bool:
    """Dispatch a structured event; return True when at least one channel confirms delivery."""
    # Resolve topic
    topic = ntfy_topic or os.environ.get("TEMPEST_NOTIFY_TOPIC")
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    webhook_url = os.environ.get("TEMPEST_WEBHOOK_URL")

    # If nothing configured, check ~/.tempest/notify.json
    config_file = Path.home() / ".tempest" / "notify.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            if not topic:
                topic = cfg.get("ntfy_topic")
            if not tg_token:
                tg_token = cfg.get("telegram_bot_token")
            if not tg_chat:
                tg_chat = cfg.get("telegram_chat_id")
            if not webhook_url:
                webhook_url = cfg.get("webhook_url")
        except Exception:
            pass

    if not topic and not (tg_token and tg_chat) and not webhook_url:
        logger.debug("No notification channel configured. Set TEMPEST_NOTIFY_TOPIC or --notify.")
        return False

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if event_type == "start":
        title = f"🚀 Tempest-NWP Run Started: {run_id}"
        body = (
            f"Forecast run initialized.\n"
            f"• Target Lead Time: {forecast_hours:.1f} hours\n"
            f"• Zulu Init Time: {run_id}\n"
            f"• Started at: {now_str}"
        )
        tags = ["rocket", "hourglass_flowing_sand"]
        priority = "default"

    elif event_type in ("da_success", "assimilation_done"):
        title = f"🎯 Tempest-NWP Variational 3D-Var DA Voltooid: {run_id}"
        n_obs = metrics.get("assimilated_obs", "N/A") if metrics else "N/A"
        dp_rms = metrics.get("delta_ps_rms", 0.0) if metrics else 0.0
        dt_rms = metrics.get("delta_t2m_rms", 0.0) if metrics else 0.0
        body = (
            f"Data Assimilatie succesvol afgerond!\n"
            f"• Geassimileerde Observaties: {n_obs}\n"
            f"• ΔPs Innovatie RMS: {dp_rms:.2f} hPa\n"
            f"• ΔT2m Innovatie RMS: {dt_rms:.2f} K\n"
            f"• Analysetijd: {now_str}"
        )
        tags = ["dart", "satellite", "chart_with_upwards_trend"]
        priority = "default"

    elif event_type in ("download_done", "download_success"):
        title = f"📥 Tempest-NWP Waarnemingen Binnengehaald: {run_id}"
        total_obs = metrics.get("total_obs", "N/A") if metrics else "N/A"
        body = (
            f"Live waarnemingen succesvol gedownload en omgezet naar WX-1!\n"
            f"• Totaal Waarnemingen: {total_obs}\n"
            f"• Venster: {forecast_hours:.1f} uur retrospectief\n"
            f"• Tijdstip: {now_str}"
        )
        tags = ["inbox_tray", "world_map"]
        priority = "default"

    elif event_type in ("gdrive_upload", "gdrive_success"):
        title = f"☁️ Tempest-NWP Google Drive Sync Voltooid: {run_id}"
        dest_str = gcs_uri or "gdrive:tempest_forecasts"
        body = (
            f"Verwachtingsbestanden succesvol gesynchroniseerd naar 5TB Google Drive!\n"
            f"• Bestemming: {dest_str}\n"
            f"• Overdrachtstijd: {runtime_sec:.1f}s\n"
            f"• Tijdstip: {now_str}"
        )
        tags = ["cloud", "floppy_disk"]
        priority = "default"

    elif event_type in ("gdrive_failure", "gdrive_error"):
        title = f"⚠️ Tempest-NWP Google Drive Upload Failed: {run_id}"
        detail = (metrics or {}).get("detail", "No upload detail returned")
        body = f"Google Drive upload/verification failed.\n• Detail: {detail}\n• Time: {now_str}"
        tags = ["warning", "cloud"]
        priority = "high"

    elif event_type == "success":
        title = f"✅ Tempest-NWP Run Finished: {run_id}"
        speedup_str = f"{forecast_hours * 3600.0 / max(runtime_sec, 1e-3):.1f}x"
        body_lines = [
            f"Forecast completed successfully!",
            f"• Simulation Time: {forecast_hours:.1f}h",
            f"• Compute Walltime: {runtime_sec:.1f}s ({speedup_str} real-time)",
            f"• Timestamp: {now_str}",
        ]
        if metrics:
            if "mslp_min" in metrics and "mslp_max" in metrics:
                body_lines.append(f"• MSLP Range: {metrics['mslp_min']:.1f} – {metrics['mslp_max']:.1f} hPa")
            if "t2m_min" in metrics and "t2m_max" in metrics:
                body_lines.append(f"• T2m Range: {metrics['t2m_min'] - 273.15:.1f}°C to {metrics['t2m_max'] - 273.15:.1f}°C")
            if "precip_max" in metrics:
                body_lines.append(f"• Max Precip: {metrics['precip_max']:.2f} mm")
        if output_dir:
            body_lines.append(f"• Local Output: {output_dir}")
        if gcs_uri:
            body_lines.append(f"• Cloud Storage: {gcs_uri}")

        body = "\n".join(body_lines)
        tags = ["white_check_mark", "partly_sunny", "satellite"]
        priority = "high"

    elif event_type == "error":
        title = f"❌ Tempest-NWP Run Failed: {run_id}"
        detail = (metrics or {}).get("detail")
        body = f"Forecast run encountered an error after {runtime_sec:.1f}s.\nDetail: {detail or 'See run log'}\nTimestamp: {now_str}"
        tags = ["x", "warning"]
        priority = "urgent"

    else:
        title = f"ℹ️ Tempest-NWP Notice: {run_id}"
        body = f"Event: {event_type} at {now_str}"
        tags = ["information_source"]
        priority = "default"

    delivery_results = []
    if topic:
        delivery_results.append(send_ntfy_notification(topic, title, body, priority=priority, tags=tags))
    if tg_token and tg_chat:
        delivery_results.append(send_telegram_notification(tg_token, tg_chat, f"*{title}*\n\n{body}"))
    if webhook_url:
        delivery_results.append(send_webhook_notification(webhook_url, title, body))
    return any(delivery_results)


def main():
    parser = argparse.ArgumentParser(description="Tempest-NWP Phone Notification Dispatcher")
    parser.add_argument("--test", action="store_true", help="Send a test notification to verify phone connectivity")
    parser.add_argument("--topic", type=str, default=None, help="ntfy.sh topic name (e.g. tempest_nwp_test)")
    parser.add_argument("--title", type=str, default="Tempest-NWP Test Notification", help="Notification title")
    parser.add_argument("--message", type=str, default="Test ping from Tempest-NWP dynamical core!", help="Message")
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        print("\nNo network request sent. Use --test to explicitly send a test notification.")
        return
    topic = args.topic or os.environ.get("TEMPEST_NOTIFY_TOPIC", "tempest_nwp_forecast")
    print(f"Sending test phone notification to ntfy topic: '{topic}'...")
    print(f"To receive this on your phone: install the free 'ntfy' app and subscribe to '{topic}'.")
    ok = send_ntfy_notification(
        topic=topic,
        title=args.title,
        message=args.message,
        priority="high",
        tags=["wave", "partly_sunny"],
    )
    if ok:
        print("Successfully delivered test push notification!")
    else:
        print("Failed to dispatch test notification.")


if __name__ == "__main__":
    main()
