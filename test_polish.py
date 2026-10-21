import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import llm


class PolishTests(unittest.TestCase):
    def test_source_counts_empty_failures_and_partial(self):
        self.assertEqual(llm.source_summary('Web', {'results': [1, 2, 3]}), 'Web — 3 results')
        self.assertEqual(llm.source_summary('Email', {'results': []}), 'Email — no recent messages')
        self.assertEqual(llm.source_summary('Docs', {'results': [1]}), 'Docs — 1 passage')
        self.assertEqual(llm.source_summary('Web', {'results': [], 'error': 'failure'}), 'Web — failed')
        self.assertEqual(llm.source_summary('Email', {'results': [1], 'error': 'failure'}), 'Email — 1 message (partial)')

    def test_footer_under_each_answer_and_rerun(self):
        with patch.object(llm, 'ask', side_effect=[iter(['Plain answer']), iter([
            {'sources': ['Web — 3 results']}, {'sources': ['Email — no recent messages']}, 'Combined answer'
        ])]):
            app = AppTest.from_file('app.py').run()
            app.chat_input[0].set_value('Hello').run()
            app.chat_input[0].set_value('Search').run()
            app.run()
            self.assertFalse(app.exception)
            answers = [m for m in app.chat_message if m.name == 'assistant']
            self.assertEqual(len(answers), 2)
            self.assertEqual(answers[0].caption[0].value, 'Sources used: None')
            self.assertEqual(answers[1].caption[0].value, 'Sources used: Web — 3 results · Email — no recent messages')

    def test_unexpected_stream_error_keeps_partial_answer_without_traceback(self):
        def stream(*args):
            yield 'Partial answer'
            yield {'sources': ['Docs — 2 passages']}
            raise RuntimeError('private internal details')
        with patch.object(llm, 'ask', side_effect=stream):
            app = AppTest.from_file('app.py').run()
            app.chat_input[0].set_value('Hi').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['messages'][-1]['content'], 'Partial answer')
            self.assertNotIn('private internal details', app.warning[0].value)
            self.assertIn('Sources used: Docs — 2 passages', [c.value for c in app.caption])

    def test_traceback_in_diagnostic_is_hidden(self):
        with patch.object(llm, 'ask', return_value=iter([{'error': 'Traceback (most recent call last):\n  File secret.py'}])):
            app = AppTest.from_file('app.py').run()
            app.chat_input[0].set_value('Hi').run()
            self.assertFalse(app.exception)
            self.assertNotIn('Traceback', app.warning[0].value)
            app.run()
            self.assertNotIn('secret.py', app.warning[0].value)


if __name__ == '__main__':
    unittest.main()
