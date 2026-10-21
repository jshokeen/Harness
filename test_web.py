import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import openai
from streamlit.testing.v1 import AppTest

import llm
import web


RESULT = {"results": [{"title": "Example", "url": "https://example.com", "content": "A verified fact"}]}


class WebTests(unittest.TestCase):
    def test_tavily_cap_request_and_complete_redacted_logging(self):
        response = httpx.Response(200, json={"results": RESULT["results"] * 7, "extra": "test-secret"},
                                  request=httpx.Request("POST", "https://api.tavily.com/search"))
        with patch.object(web, "get_config", return_value="test-secret"), patch.object(web.httpx, "post", return_value=response) as post, patch.object(web, "logger") as log:
            result = web.search("current facts")
            self.assertEqual(len(result["results"]), 5)
            self.assertEqual(post.call_args.kwargs["json"]["max_results"], 5)
            self.assertFalse(post.call_args.kwargs["json"]["include_raw_content"])
            self.assertEqual(log.info.call_count, 2)
            self.assertNotIn("test-secret", str(log.mock_calls))
            self.assertIn("extra", str(log.mock_calls))

    def test_tavily_failure_and_missing_key(self):
        with patch.object(web, "get_config", return_value=""), patch.object(web.httpx, "post") as post:
            self.assertIn("unavailable", web.search("test")["error"])
            post.assert_not_called()
        response = httpx.Response(429, text="full raw failure", request=httpx.Request("POST", "https://api.tavily.com/search"))
        with patch.object(web, "get_config", return_value="dummy"), patch.object(web.httpx, "post", return_value=response), patch.object(web, "logger") as log:
            self.assertIn("failed", web.search("test")["error"])
            self.assertIn("429", str(log.error.call_args))
            self.assertIn("full raw failure", str(log.error.call_args))

    def compatible(self, provider, enabled=True, key=True, call_tool=True, malformed=False, fail_search=False, reject=False):
        requests = []
        def response(request):
            payload = json.loads(request.content)
            requests.append(payload)
            use_tool = enabled and key and call_tool and len(requests) == 1
            if use_tool and reject:
                return httpx.Response(400, json={"error": {"message": "tool schema rejected", "type": "invalid_request_error"}})
            deltas = ([{"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": '{"query":'}}]},
                       {"tool_calls": [{"index": 0, "function": {"arguments": 'bad' if malformed else '"news"}'}}]}]
                      if use_tool else [{"content": "Grounded "}, {"content": "answer"}])
            events = []
            for delta in deltas + [{}]:
                events.append("data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0,
                    "model": llm.PROVIDERS[provider][0], "choices": [{"index": 0, "delta": delta,
                    "finish_reason": ("tool_calls" if use_tool else "stop") if not delta else None}]}) + "\n\n")
            return httpx.Response(200, text="".join(events) + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        real_client = openai.OpenAI
        with patch.object(llm, "get_config", side_effect=lambda name: "dummy" if name != "TAVILY_API_KEY" or key else ""), patch.object(
            llm.openai, "OpenAI", side_effect=lambda **kw: real_client(**kw, http_client=httpx.Client(transport=httpx.MockTransport(response)))
        ), patch.object(web, "search", return_value={"error": "Web search failed.", "results": []} if fail_search else RESULT) as search:
            events = list(llm.ask(provider, [{"role": "user", "content": "news"}], web_enabled=enabled))
            self.assertEqual(search.call_count, int(enabled and key and call_tool))
        self.assertEqual("".join(x for x in events if isinstance(x, str)), "Grounded answer")
        if enabled and key and call_tool:
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]["tool_choice"], "auto")
            self.assertNotIn("tools", requests[1])
            self.assertIn("results", str(requests[1]["messages"]))
            if not malformed and not reject:
                self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "call_1")
            self.assertLess(next(i for i,x in enumerate(events) if isinstance(x,dict) and "status" in x), events.index("Grounded "))
        else:
            self.assertFalse(any(isinstance(x, dict) and "sources" in x for x in events))
        return requests, events

    def test_openai_and_grok_tool_roundtrip(self):
        for provider in ("OpenAI", "Grok"):
            with self.subTest(provider=provider):
                self.compatible(provider)

    def test_toggle_off_missing_key_and_model_declining(self):
        for provider in ("OpenAI", "Grok"):
            for options in ({"enabled": False}, {"key": False}, {"call_tool": False}):
                with self.subTest(provider=provider, options=options):
                    requests, _ = self.compatible(provider, **options)
                    if options != {"call_tool": False}:
                        self.assertNotIn("tools", requests[0])

    def test_grok_malformed_and_rejected_tool_fallback(self):
        self.compatible("Grok", malformed=True)
        self.compatible("Grok", reject=True)

    def test_search_failure_still_answers(self):
        for provider in ("OpenAI", "Grok"):
            _, events = self.compatible(provider, fail_search=True)
            self.assertIn({"sources": ["Web — failed"]}, events)

    def test_claude_tool_result_protocol_and_gate(self):
        for enabled in (True, False):
            tool_message = anthropic.types.Message(id="msg", type="message", role="assistant", model=llm.MODEL,
                content=[{"type": "tool_use", "id": "call_1", "name": "web_search", "input": {"query": "news"}}],
                stop_reason="tool_use", usage={"input_tokens": 1, "output_tokens": 1})
            answer = SimpleNamespace(stop_reason="end_turn")
            first, second = MagicMock(), MagicMock()
            first.text_stream = iter([] if enabled else ["Answer"])
            first.get_final_message.return_value = tool_message if enabled else answer
            second.text_stream = iter(["Answer"])
            second.get_final_message.return_value = answer
            with patch.object(llm, "get_config", return_value="dummy"), patch.object(llm.anthropic, "Anthropic") as factory, patch.object(web, "search", return_value=RESULT) as search:
                client = factory.return_value.__enter__.return_value
                contexts = [MagicMock(), MagicMock()]
                contexts[0].__enter__.return_value = first
                contexts[1].__enter__.return_value = second
                client.messages.stream.side_effect = contexts
                events = list(llm.ask("Claude", [{"role": "user", "content": "news"}], web_enabled=enabled))
                self.assertIn("Answer", events)
                self.assertEqual(search.call_count, int(enabled))
                if enabled:
                    request = client.messages.stream.call_args.kwargs
                    self.assertEqual(request["messages"][-1]["content"][0]["tool_use_id"], "call_1")
                    self.assertNotIn("tools", request)
                else:
                    self.assertNotIn("tools", client.messages.stream.call_args.kwargs)

    def test_status_is_yielded_before_search_starts(self):
        with patch.object(web, "search", return_value=RESULT) as search:
            stream = llm._execute("web_search", {"query": "news"}, True)
            self.assertIn("status", next(stream))
            search.assert_not_called()
            list(stream)
            search.assert_called_once()

    def test_ui_status_sources_persist_and_no_traceback(self):
        with patch.object(llm, "ask", return_value=iter([
            {"status": "🌐 Searching the web…"}, {"sources": ["Web"]}, "Answer"
        ])):
            app = AppTest.from_file("app.py").run()
            app.toggle[0].set_value(True).run()
            app.chat_input[0].set_value("news").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["messages"][-1]["sources"], ["Web"])
            app.run()
            self.assertTrue(any(c.value == "Sources used: Web" for c in app.caption))


if __name__ == "__main__":
    unittest.main()
