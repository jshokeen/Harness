# The Harness

Streaming chat with Claude (`claude-sonnet-4-6`), OpenAI (`gpt-4o`),
and Grok (`grok-2-latest`, via `https://api.x.ai/v1`).

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py --server.address localhost
```

Open http://localhost:8501. Add any of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
or `XAI_API_KEY` and select the corresponding provider.
Use Settings to change the sidebar title and enter credentials for the session.
Blank credential fields preserve existing settings. Environment variables (or
an optional local `.env`) provide fallback values; secrets are never prefilled.
Enable Web and supply `TAVILY_API_KEY` to allow model-selected Tavily search.
One search per reply returns at most five snippets. The app shows search progress
and a Sources used footer under every answer. Footers show actual result counts,
empty/failed outcomes, or None when no capability ran, and persist across reruns.
Grok falls back to direct search if tool calls are rejected or malformed.
Upload one `.txt` or `.pdf` and enable Docs for model-selected keyword retrieval.
Documents stay in session memory: 800-character chunks, 100-character overlap,
up to five matching passages injected as tool context. Web and Docs may both run.
Extraction caps text at 200,000 characters and PDFs at 100 pages (10 MB maximum).
Scanned/empty, corrupt, and locked PDFs show a short message. No OCR or vector DB.
Enable Email with `GMAIL_USER` and `GMAIL_APP_PASSWORD` to allow model-selected
read-only Gmail access. Password spaces/invisible characters are removed.
The latest ten messages are fetched with BODY.PEEK from a read-only INBOX,
falling back to All Mail only when INBOX is empty. Bodies are limited to 1,000
characters in model context. No mailbox searches, writes, or read-flag changes.
IMAP commands and full results (including fetched email contents) are logged
locally to console and ignored `harness.log`; credentials are redacted.
Missing keys and failed calls show short inline messages, preserving partial replies.
Provider call starts, successes, and full failure diagnostics go to the console and
ignored `harness.log`; credential values are excluded.
Full Tavily request payloads and responses are logged, excluding credentials.
UI exception details are disabled; unexpected streaming failures preserve partial
answers and show a short message while full error diagnostics remain in the log.

Optional Sheets logging: enable the Google Sheets API for your service-account
project, share the target spreadsheet with the service account's `client_email`
as an editor, and set `GOOGLE_SERVICE_ACCOUNT_JSON` and `TARGET_SHEET_ID` in
Settings or the environment. No browser sign-in is used.
Each completed chat turn appends one row to the first worksheet's A:G table:
UTC timestamp, provider, model, user message, assistant reply, sources, diagnostic.
You may add these column headers yourself. Values are written as literal text.
Appends run in a background queue and never delay replies. Missing configuration
skips silently; failures only log. Queued rows are in memory (up to 100) and can
be lost if the app process exits; failed appends are not retried to avoid duplicates.

## Environment variables

All credentials are optional individually. At least one provider key is needed
for real chat. Docs needs no additional key. Settings overrides environment
values for the current browser session only.

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude |
| `OPENAI_API_KEY` | OpenAI |
| `XAI_API_KEY` | Grok |
| `TAVILY_API_KEY` | Web search |
| `GMAIL_USER` | Gmail address |
| `GMAIL_APP_PASSWORD` | Google app password; spaces and invisible characters stripped |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Complete service-account JSON string for optional Sheets logging |
| `TARGET_SHEET_ID` | Spreadsheet ID for optional Sheets logging |

Use `.env.example` as a local template. `.env`, JSON credential files, private
keys, logs, and Streamlit secrets are excluded from Git.

## Render setup

Connect this repository as a Python web service, using `main` and these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Health check: `/_stcore/health`
- Python version: `3.12.10` (set `PYTHON_VERSION`)

The same settings are provided in `render.yaml`; automatic deployment is disabled.
Render supplies `PORT`. Add the credentials you need in Render's Environment
settings, never to the repository. Leave unused credentials unset.

Real-key checks remain pending; see [LOCAL_CHECKPOINT.md](LOCAL_CHECKPOINT.md).
Verify those before deploying. Hosting must permit outbound Gmail IMAP on port
993 for Email to work. The app has no sign-in/access restriction, so anyone who
can access a hosted instance can use its environment-configured capabilities.
Choose access restrictions before hosting private Gmail or document data.
