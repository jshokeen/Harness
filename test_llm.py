"""Local verification without paid API calls: python -m unittest -v."""
import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
from streamlit.testing.v1 import AppTest

import config
import llm


class ClaudeTests(unittest.TestCase):
    def test_config_session_precedence_and_sanitization(self):
        state = {}
        with patch.object(config, "st", SimpleNamespace(session_state=state)), patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "  env-\u00a0key\u200b  "}, clear=True
        ):
            self.assertEqual(config.get_config("ANTHROPIC_API_KEY"), "env-key")
            config.save_settings("Harness", {"ANTHROPIC_API_KEY": " ui-\u00a0key ",
                                               "GMAIL_APP_PASSWORD": "abcd efgh\u00a0ijkl mnop"})
            self.assertEqual(config.get_config("ANTHROPIC_API_KEY"), "ui-key")
            self.assertEqual(config.get_config("GMAIL_APP_PASSWORD"), "abcdefghijklmnop")
            self.assertEqual(config.get_config("XAI_API_KEY"), "")
            self.assertNotEqual(os.environ["ANTHROPIC_API_KEY"], "ui-key")

    def test_missing_key_and_unconnected_provider_never_call_api(self):
        with patch.object(llm, "get_config", return_value=""), patch.object(llm.anthropic, "Anthropic") as client:
            self.assertIn("unavailable", list(llm.ask("Claude", []))[0]["error"])
            self.assertIn("unavailable", list(llm.ask("Grok", []))[0]["error"])
            self.assertIn("unavailable", list(llm.ask("OpenAI", []))[0]["error"])
            client.assert_not_called()

    def test_stream_is_incremental_and_history_is_clean(self):
        stream = MagicMock()
        stream.text_stream = iter(["Hello", " there"])
        stream.get_final_message.return_value = SimpleNamespace(stop_reason="end_turn")
        with patch.object(llm, "get_config", return_value="dummy"), patch.object(llm.anthropic, "Anthropic") as factory:
            client = factory.return_value.__enter__.return_value
            client.messages.stream.return_value.__enter__.return_value = stream
            history = [{"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "", "error": "prior error"},
                       {"role": "user", "content": "Again"}]
            result = llm.ask("Claude", history)
            self.assertEqual(next(result), "Hello")
            self.assertEqual(next(result), " there")
            self.assertEqual(list(result), [])
            args = client.messages.stream.call_args.kwargs
            self.assertEqual(args["model"], "claude-sonnet-4-6")
            self.assertEqual(args["messages"], [history[0], history[2]])

    def test_real_sdk_http_failure_is_logged_and_safe(self):
        real_client = anthropic.Anthropic
        transport = httpx.MockTransport(lambda request: httpx.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid dummy-key"}}
        ))
        def client_factory(**kwargs):
            return real_client(**kwargs, http_client=httpx.Client(transport=transport))
        with patch.object(llm, "get_config", return_value="dummy-key"), patch.object(
            llm.anthropic, "Anthropic", side_effect=client_factory
        ), patch.object(llm, "logger") as log:
            result = list(llm.ask("Claude", [{"role": "user", "content": "Hi"}]))
            self.assertIn("authenticate", result[0]["error"])
            diagnostic = log.error.call_args.args[1]
            self.assertIn("401", diagnostic)
            self.assertIn("raw_response=", diagnostic)
            self.assertIn("authentication_error", diagnostic)
            self.assertNotIn("dummy-key", diagnostic)

    def test_midstream_failure_preserves_text(self):
        def interrupted():
            yield "Partial answer"
            raise RuntimeError("connection interrupted")
        stream = MagicMock()
        stream.text_stream = interrupted()
        with patch.object(llm, "get_config", return_value="dummy"), patch.object(llm.anthropic, "Anthropic") as factory:
            factory.return_value.__enter__.return_value.messages.stream.return_value.__enter__.return_value = stream
            result = list(llm.ask("Claude", [{"role": "user", "content": "Hi"}]))
            self.assertEqual(result[0], "Partial answer")
            self.assertIn("could not finish", result[1]["error"])

    def test_app_stream_and_missing_key(self):
        with patch.object(llm, "get_config", return_value=""):
            app = AppTest.from_file("app.py").run()
            app.chat_input[0].set_value("Hello").run()
            self.assertFalse(app.exception)
            self.assertIn("unavailable", app.warning[0].value)
        with patch.object(llm, "ask", return_value=iter(["Streamed", " reply"])):
            app = AppTest.from_file("app.py").run()
            app.chat_input[0].set_value("Hello").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["messages"][-1]["content"], "Streamed reply")


class OpenAICompatibleTests(unittest.TestCase):
    def run_stream(self, provider, *, status=200, complete=True):
        model, key = llm.PROVIDERS[provider]
        host = "api.x.ai" if provider == "Grok" else "api.openai.com"
        history = [{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "Hello"},
                   {"role": "user", "content": "Continue"}]
        def respond(request):
            self.assertEqual(request.url.host, host)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer selected-dummy")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], model)
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["messages"], history)
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "failure selected-dummy", "type": "test_error"}})
            events = []
            for content, finish in [(None, None), ("Hello", None), (" again", None), (None, "stop")]:
                if finish and not complete:
                    continue
                chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0,
                         "model": model, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}
                events.append("data: " + json.dumps(chunk) + "\n\n")
            events.append("data: [DONE]\n\n")
            return httpx.Response(200, text="".join(events), headers={"content-type": "text/event-stream"})
        real_client = openai.OpenAI
        def client_factory(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            return real_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        with patch.object(llm, "get_config", side_effect=lambda name: "selected-dummy" if name == key else ""), patch.object(
            llm.openai, "OpenAI", side_effect=client_factory
        ), patch.object(llm, "logger") as log:
            result = list(llm.ask(provider, history))
            if status != 200 or not complete:
                detail = log.error.call_args.args[1]
                self.assertIn(provider, detail)
                self.assertNotIn("selected-dummy", detail)
                if status != 200:
                    self.assertIn(str(status), detail)
                    self.assertIn("raw_response=", detail)
            else:
                self.assertEqual(log.info.call_count, 2)
            return result

    def test_both_provider_streams_use_correct_endpoint_key_and_history(self):
        for provider in ("OpenAI", "Grok"):
            with self.subTest(provider=provider):
                self.assertEqual(self.run_stream(provider), ["Hello", " again"])

    def test_both_provider_errors_are_safe(self):
        for provider in ("OpenAI", "Grok"):
            for status in (401, 429, 500):
                with self.subTest(provider=provider, status=status):
                    result = self.run_stream(provider, status=status)
                    self.assertIn(provider, result[0]["error"])
                    self.assertNotIn("selected-dummy", str(result))

    def test_incomplete_stream_keeps_partial_reply(self):
        for provider in ("OpenAI", "Grok"):
            result = self.run_stream(provider, complete=False)
            self.assertEqual(result[:2], ["Hello", " again"])
            self.assertIn("could not finish", result[-1]["error"])

    def test_ui_selects_each_provider_without_keys(self):
        with patch.object(llm, "get_config", return_value=""):
            for provider in llm.PROVIDERS:
                with self.subTest(provider=provider):
                    app = AppTest.from_file("app.py").run()
                    app.selectbox[0].select(provider).run()
                    app.chat_input[0].set_value("Hi").run()
                    self.assertFalse(app.exception)
                    self.assertIn(provider + " is unavailable", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
