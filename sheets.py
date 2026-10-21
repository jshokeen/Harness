"""Best-effort, in-memory background Sheets appends; never access Streamlit in the worker."""
from datetime import datetime, timezone
import json
from queue import Queue, Full
import re
from threading import Lock, Thread
from urllib.parse import parse_qs, quote

import httpx
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from config import CREDENTIAL_GROUPS, get_config
from diagnostics import get_logger

logger = get_logger()
_jobs = Queue(maxsize=100)
_lock = Lock()
_worker = None


def _redact(value, secrets):
    text = str(value)
    for secret in sorted((str(s) for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
        text = text.replace(json.dumps(secret)[1:-1], "[REDACTED]")
    return re.sub(r'("(?:access_token|refresh_token|assertion)"\s*:\s*")[^"]*', r'\1[REDACTED]', text)


def _append(service_json, sheet_id, row, secrets):
    secrets = list(secrets)
    response = None
    try:
        info = json.loads(service_json)
        secrets.extend(str(value) for value in info.values())
        # Credentials must only be exchanged with Google's token endpoint.
        info["token_uri"] = "https://oauth2.googleapis.com/token"
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        with requests.Session() as session:
            transport = Request(session=session)

            def auth_request(url, method="GET", body=None, headers=None, **kwargs):
                if body:
                    encoded = body.decode() if isinstance(body, bytes) else body
                    secrets.extend(parse_qs(encoded).get("assertion", []))
                logger.info("Sheets authentication call started")
                try:
                    result = transport(url=url, method=method, body=body, headers=headers, timeout=30)
                    if result.status >= 400:
                        logger.error("Sheets authentication call failed: status=%s raw_response=%s", result.status, _redact(result.data, secrets))
                    else:
                        logger.info("Sheets authentication call succeeded: status=%s", result.status)
                    return result
                except Exception as error:
                    logger.error("Sheets authentication call failed: %s", _redact(f"{type(error).__name__}: {error!s}; {error!r}", secrets))
                    raise

            credentials.refresh(auth_request)
        secrets.append(credentials.token)
        logger.info("Sheets append call started: one conversation row")
        response = httpx.post(
            f"https://sheets.googleapis.com/v4/spreadsheets/{quote(sheet_id, safe='')}/values/A%3AG:append",
            headers={"Authorization": f"Bearer {credentials.token}"},
            params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
            json={"majorDimension": "ROWS", "values": [row]}, timeout=30,
        )
        response.raise_for_status()
        logger.info("Sheets append call succeeded: status=%s result=%s", response.status_code, _redact(response.text, secrets))
    except Exception as error:
        logger.error("Sheets background append failed: %s", _redact(
            f"status={response.status_code if response is not None else None} "
            f"exception={type(error).__name__}: {error!s}; {error!r}\n"
            f"raw_response={response.text if response is not None else None}", secrets))


def _run():
    while True:
        job = _jobs.get()
        try:
            _append(*job)
        except Exception as error:
            # Keep the queue alive even if an unexpected worker failure occurs.
            logger.error("Sheets worker failed: %s", _redact(f"{type(error).__name__}: {error}", job[3]))
        finally:
            _jobs.task_done()
            del job


def log_conversation(provider, model, prompt, reply):
    """Snapshot session credentials and enqueue once, after a reply is rendered."""
    global _worker
    service_json, sheet_id, secrets = "", "", ()
    try:
        service_json = get_config("GOOGLE_SERVICE_ACCOUNT_JSON")
        sheet_id = get_config("TARGET_SHEET_ID")
        if not service_json or not sheet_id:
            logger.info("Sheets skipped: service account or target sheet unavailable")
            return
        secrets = tuple(get_config(key) for keys in CREDENTIAL_GROUPS.values() for key in keys)
        row = [datetime.now(timezone.utc).isoformat(), provider, model, prompt,
               reply.get("content", ""), " · ".join(reply.get("sources") or ["None"]), reply.get("error", "")]
        with _lock:
            if _worker is None or not _worker.is_alive():
                _worker = Thread(target=_run, name="harness-sheets", daemon=True)
                _worker.start()
        _jobs.put_nowait((service_json, sheet_id, row, secrets))
        logger.info("Sheets conversation row queued")
    except Full:
        logger.warning("Sheets skipped: background queue is full")
    except Exception as error:
        # Avoid ever turning optional logging into a chat failure.
        logger.error("Sheets enqueue failed: %s", _redact(f"{type(error).__name__}: {error!s}; {error!r}", (*secrets, service_json, sheet_id)))
