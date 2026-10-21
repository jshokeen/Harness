# Localhost checkpoint — 21 October 2026

Local checks passed. Live service verification is pending; the user requested
local checks only. Nothing has been deployed.

- App health: `http://localhost:8501/_stcore/health` returned `ok`.
- Dependency check: no broken requirements.
- 44 automated tests passed across the provider, Web, RAG, Gmail, polish,
  Sheets, and combined-capability suites. External services were mocked.
- Combined Web + Docs + Email context reaches the final answer for all three
  provider paths; each tool runs once and the footer reflects its result count.
- Browser upload: a real local TXT fixture extracted successfully.
- Earlier browser checks also covered text PDFs and PDFs without extractable text.
- The app reports missing provider, Tavily, and Gmail credentials without crashing.

## Live checks still required before deployment

Enter credentials in Settings and save for the session. Keep the same browser
session open. Never paste secrets into chat or commit them.

| Check | Action | Expected |
| --- | --- | --- |
| Claude | Select Claude; all tools off; ask “Reply with hello.” | Streamed reply; Sources used: None |
| OpenAI | Repeat with OpenAI | Same, using gpt-4o |
| Grok | Repeat with Grok | Same, using grok-2-latest |
| Web | Enable Web; ask for current news with source links | Search status, grounded reply, Web result count |
| Docs | Upload a TXT/PDF containing a known fact; enable Docs and ask about it | Reading status, correct fact, passage count |
| Email | Enable Email; ask to summarize recent inbox messages | Checking status, up to ten messages or an empty result |
| Combined | Enable all three; ask to compare a document fact with recent email and current web information | Relevant tools run; footer lists those that fired |
| Optional Sheets | Configure service account + shared target sheet, then send one turn | One row appended; rerunning the UI adds no duplicate |

Repeat Web, Docs, and Email checks with each provider. Verify Gmail messages
remain unread. If any real API rejects a configured model or credential, inspect
the full failure in `harness.log` and resolve it before deployment. Do not infer
live service success from the mocked tests.
