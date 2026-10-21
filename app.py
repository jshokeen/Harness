import hashlib
import streamlit as st
import rag
import sheets

from config import CREDENTIAL_GROUPS, get_config, save_settings
from diagnostics import get_logger
from llm import PROVIDERS, ask
from web import redact

st.set_page_config(page_title="The Harness", page_icon="✳", layout="centered")
logger = get_logger()
st.set_option("client.showErrorDetails", "none")


def safe_notice(value):
    text = redact(str(value))
    if "Traceback (most recent call last)" in text or "\n  File " in text:
        logger.error("UI diagnostic suppressed: %s", text)
        return "Something went wrong. Details are in harness.log; please try again."
    return text


st.markdown("""
<style>
html, body {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
[data-testid="stSidebar"] {border-right: 1px solid #E2E8F0;}
[data-testid="stSidebar"] h1 {font-size: 2rem;}
[data-testid="stChatMessage"] {border: 1px solid #E2E8F0; border-radius: 16px;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: 90%;
    background: #F8FAFC;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-right: auto; max-width: 95%; background: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("harness_title", "The Harness")
st.session_state.setdefault("messages", [])


@st.dialog("Settings", width="large")
def settings():
    st.caption("Saved values apply only to this session. Blank fields keep existing values.")
    with st.form("settings_form"):
        title = st.text_input("Harness title", value=st.session_state.harness_title)
        values = {}
        for group, keys in CREDENTIAL_GROUPS.items():
            st.subheader(group)
            for key in keys:
                status = "● set" if get_config(key) else "○ empty"
                label = f"{key} · {status}"
                if key == "GOOGLE_SERVICE_ACCOUNT_JSON":
                    # Password input also accepts pasted JSON while keeping it masked.
                    values[key] = st.text_input(label, type="password", help="Paste the complete service-account JSON.")
                    uploaded_json = st.file_uploader("Or upload service-account JSON", type=["json"])
                    if uploaded_json is not None:
                        try:
                            values[key] = uploaded_json.getvalue().decode("utf-8-sig")
                        except UnicodeDecodeError:
                            st.warning("Please upload a UTF-8 JSON file.")
                else:
                    values[key] = st.text_input(
                        label,
                        type="default" if key in ("GMAIL_USER", "TARGET_SHEET_ID") else "password",
                    )
        if st.form_submit_button("Save for this session", type="primary"):
            save_settings(title, values)
            logger.info("Session settings saved (values omitted)")
            st.rerun()


with st.sidebar:
    st.title(st.session_state.harness_title)
    st.subheader("How it works")
    st.markdown("Pick a model.  \nToggle Web, Docs, or Gmail.  \nAsk a question.  \nGround answers in real sources.")
    st.divider()
    st.caption("STEP 8 · Conversation logging\n\nWeb, documents, and recent email in one chat.")
    for name, (_, key) in PROVIDERS.items():
        st.caption(name + " · " + ("● available" if get_config(key) else "○ unavailable — add a key in Settings"))
    st.caption("Web · " + ("● available" if get_config("TAVILY_API_KEY") else "○ unavailable — add a key in Settings"))
    st.caption("Email · " + ("● available" if get_config("GMAIL_USER") and get_config("GMAIL_APP_PASSWORD") else "○ unavailable — add credentials in Settings"))

heading, settings_button = st.columns([3, 1])
with heading:
    st.title("Chat")
with settings_button:
    if st.button("⚙️ Settings", use_container_width=True):
        settings()
st.caption("A focused space for questions, documents, and ideas.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["content"]:
            st.write(message["content"])
        if message.get("error"):
            st.warning(safe_notice(message["error"]))
        if message["role"] == "assistant":
            st.caption("Sources used: " + " · ".join(message.get("sources") or ["None"]))

with st.bottom.container(border=True):
    model, web, gmail, docs = st.columns([2, 1, 1, 1], vertical_alignment="center")
    with model:
        provider = st.selectbox("Model", ["Claude", "OpenAI", "Grok"], label_visibility="collapsed")
    with web:
        web_enabled = st.toggle("🌐 Web")
    with gmail:
        email_enabled = st.toggle("✉️ Email")
    with docs:
        docs_enabled = st.toggle("📄 Docs")
    uploaded_doc = st.file_uploader("Attach a document", type=["txt", "pdf"], label_visibility="collapsed")
    if uploaded_doc is None:
        st.session_state.pop("document", None)
    else:
        data = uploaded_doc.getvalue()
        identity = (uploaded_doc.name, hashlib.sha256(data).hexdigest())
        if st.session_state.get("document", {}).get("identity") != identity:
            st.session_state["document"] = {"identity": identity, "chunks": [], "processed": False}
        if docs_enabled and not st.session_state.document["processed"]:
            with st.spinner("📄 Reading your document…"):
                st.session_state.document.update(rag.extract(uploaded_doc.name, data), processed=True)
        if st.session_state.document.get("notice"):
            st.caption(st.session_state.document["notice"])
        elif st.session_state.document["chunks"]:
            st.caption(f"Document ready · {len(st.session_state.document['chunks'])} chunks")
        else:
            st.caption("Enable Docs to read this document.")
    prompt = st.chat_input("Ask anything…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    reply = {"role": "assistant", "content": ""}
    with st.chat_message("assistant"):
        status = st.empty()
        output = st.empty()
        try:
            for chunk in ask(provider, st.session_state.messages, web_enabled,
                             docs_enabled, st.session_state.get("document", {}).get("chunks", []), email_enabled):
                if isinstance(chunk, dict):
                    if "status" in chunk:
                        status.caption(chunk["status"])
                    elif "sources" in chunk:
                        reply["sources"] = list(dict.fromkeys(reply.get("sources", []) + chunk["sources"]))
                        status.empty()
                    elif "error" in chunk:
                        reply["error"] = chunk["error"]
                        st.warning(safe_notice(chunk["error"]))
                else:
                    reply["content"] += chunk
                    output.markdown(reply["content"] + " ▌")
        except Exception as error:
            logger.error("Chat rendering failed: %s", redact(f"{type(error).__name__}: {error!s}; exception={error!r}"))
            reply["error"] = "Could not finish this reply. Please try again. Details are in harness.log."
            st.warning(reply["error"])
        if reply["content"]:
            output.markdown(reply["content"])
        status.empty()
        st.caption("Sources used: " + " · ".join(reply.get("sources") or ["None"]))
    st.session_state.messages.append(reply)
    sheets.log_conversation(provider, PROVIDERS[provider][0], prompt, reply)
