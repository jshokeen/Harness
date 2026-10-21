"""Resolve session overrides without exposing or persisting credentials."""
import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CREDENTIAL_GROUPS = {
    "Models": ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"),
    "Web": ("TAVILY_API_KEY",),
    "Email": ("GMAIL_USER", "GMAIL_APP_PASSWORD"),
    "Sheets": ("GOOGLE_SERVICE_ACCOUNT_JSON", "TARGET_SHEET_ID"),
}


def clean_value(key, value):
    value = str(value or "").strip()
    if key == "GMAIL_APP_PASSWORD":
        return re.sub(r"[^a-zA-Z0-9]", "", value)
    if key != "GOOGLE_SERVICE_ACCOUNT_JSON":
        return "".join(char for char in value if ord(char) < 128).strip()
    return value


def get_config(key):
    overrides = st.session_state.get("credential_overrides", {})
    return clean_value(key, overrides.get(key, os.environ.get(key, "")))


def save_settings(title, values):
    # These keys are deliberately separate from widget keys. Keep secrets scoped
    # to this session; os.environ is process-wide and leaks between sessions.
    overrides = dict(st.session_state.get("credential_overrides", {}))
    for key, value in values.items():
        if value.strip():
            overrides[key] = clean_value(key, value)
    st.session_state["credential_overrides"] = overrides
    st.session_state["harness_title"] = title.strip() or "The Harness"
