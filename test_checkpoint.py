"""Local integration checkpoint: all three capabilities in one conversation."""
import unittest
from unittest.mock import patch

import anthropic
import gmail
import llm


class CheckpointTests(unittest.TestCase):
    def test_all_three_capabilities_then_answer_for_each_provider(self):
        for provider in llm.PROVIDERS:
            with self.subTest(provider=provider):
                turns = []

                def turn(*args, **kwargs):
                    history, allowed = (args[1], args[2]) if provider == 'Claude' else (args[3], args[4])
                    turns.append(set(allowed))
                    if allowed:
                        name = next(name for name in ('web_search', 'read_document', 'read_email') if name in allowed)
                        if provider == 'Claude':
                            return anthropic.types.Message(id='msg', type='message', role='assistant', model=llm.MODEL,
                                content=[{'type':'tool_use','id':name,'name':name,'input':{'query':'budget'}}],
                                stop_reason='tool_use', usage={'input_tokens':1,'output_tokens':1})
                        return '', [{'id':name,'type':'function','function':{'name':name,'arguments':'{"query":"budget"}'}}]
                    for fact in ('web fact', 'budget 42', 'email fact'):
                        self.assertIn(fact, str(history))
                    yield 'Combined answer'
                    return type('Final', (), {'stop_reason':'end_turn'})() if provider == 'Claude' else ('Combined answer', [])

                helper = '_claude_turn' if provider == 'Claude' else '_compatible_turn'
                with patch.object(llm, 'get_config', return_value='dummy'), patch.object(llm, helper, side_effect=turn), patch.object(llm.importlib, 'reload'), patch.object(llm.web, 'search', return_value={'results':[{'content':'web fact'}]}) as web, patch.object(gmail, 'fetch_recent', return_value={'results':[{'body':'email fact'}]}) as email:
                    events = list(llm.ask(provider, [{'role':'user','content':'Compare the budgets'}], True, True, ['budget 42'], True))
                self.assertEqual([len(allowed) for allowed in turns], [3, 2, 1, 0])
                self.assertEqual(events[-1], 'Combined answer')
                self.assertEqual([event['sources'][0] for event in events if isinstance(event, dict) and 'sources' in event],
                                 ['Web — 1 result', 'Docs — 1 passage', 'Email — 1 message'])
                web.assert_called_once()
                email.assert_called_once()


if __name__ == '__main__':
    unittest.main()
