import io
import unittest
from unittest.mock import patch

import anthropic
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

import llm
import rag


def pdf_bytes(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 20 200 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class RagTests(unittest.TestCase):
    def test_chunk_size_overlap_and_tail(self):
        text = ''.join(chr(65 + i % 26) for i in range(1601))
        chunks = rag.chunk_text(text)
        self.assertEqual([len(c) for c in chunks], [800, 800, 201])
        self.assertEqual(chunks[0][-100:], chunks[1][:100])
        self.assertEqual(chunks[0] + ''.join(c[100:] for c in chunks[1:]), text)

    def test_text_pdf_blank_corrupt_and_truncation(self):
        self.assertEqual(rag.extract('a.txt', 'café budget'.encode())['chunks'], ['café budget'])
        self.assertIn('Budget is 42', ''.join(rag.extract('a.pdf', pdf_bytes('Budget is 42'))['chunks']))
        self.assertIn('no extractable text', rag.extract('scan.pdf', pdf_bytes())['notice'])
        self.assertEqual(rag.extract('bad.pdf', b'not a pdf')['chunks'], [])
        self.assertIn('empty', rag.extract('a.txt', b'')['notice'])
        large = rag.extract('big.txt', b'x' * (rag.MAX_CHARS + 1))
        self.assertIn('truncated', large['notice'])
        self.assertLessEqual(len(large['chunks'][0]) + sum(len(c)-100 for c in large['chunks'][1:]), rag.MAX_CHARS)

    def test_keyword_ranking_and_no_match(self):
        chunks = ['Launch details', 'Budget budget', 'Launch budget is 42']
        self.assertEqual(rag.retrieve('launch budget', chunks)['results'][0]['chunk'], 3)
        self.assertEqual(rag.retrieve('penguins', chunks)['results'], [])
        self.assertEqual(len(rag.retrieve('summarize my document', chunks)['results']), 3)

    def test_docs_status_precedes_retrieval_and_off_gate(self):
        with patch.object(rag, 'retrieve', return_value={'results': []}) as retrieve:
            stream = llm._execute('read_document', {'query': 'budget'}, True, ['budget 42'])
            self.assertEqual(next(stream), {'status': '📄 Reading your document…'})
            retrieve.assert_not_called()
            self.assertIn({'sources': ['Docs — no matching passages']}, list(stream))
            retrieve.reset_mock()
            list(llm._execute('read_document', {'query': 'budget'}, False, ['budget 42']))
            retrieve.assert_not_called()

    def test_all_providers_docs_context_and_combined_sources(self):
        for provider in llm.PROVIDERS:
            for combined in (False, True):
                with self.subTest(provider=provider, combined=combined):
                    seen = []
                    def turn(*args, **kwargs):
                        history, allowed = (args[1], args[2]) if provider == 'Claude' else (args[3], args[4])
                        seen.append(set(allowed))
                        name = 'read_document' if 'read_document' in allowed else 'web_search' if 'web_search' in allowed else None
                        if name:
                            if provider == 'Claude':
                                return anthropic.types.Message(id='msg', type='message', role='assistant', model=llm.MODEL,
                                    content=[{'type':'tool_use','id':name,'name':name,'input':{'query':'budget'}}],
                                    stop_reason='tool_use', usage={'input_tokens':1,'output_tokens':1})
                            return '', [{'id':name,'type':'function','function':{'name':name,'arguments':'{"query":"budget"}'}}]
                        self.assertIn('budget is 42', str(history))
                        yield 'Answer'
                        if provider == 'Claude':
                            return type('Final', (), {'stop_reason':'end_turn'})()
                        return 'Answer', []
                    helper = '_claude_turn' if provider == 'Claude' else '_compatible_turn'
                    factory = llm.anthropic if provider == 'Claude' else llm.openai
                    cls = 'Anthropic' if provider == 'Claude' else 'OpenAI'
                    with patch.object(llm,'get_config',return_value='dummy'), patch.object(factory,cls), patch.object(llm,helper,side_effect=turn), patch.object(llm.web,'search',return_value={'results':[{'content':'web fact'}]}):
                        events = list(llm.ask(provider,[{'role':'user','content':'budget'}],combined,True,['budget is 42']))
                    self.assertIn({'sources':['Docs — 1 passage']},events)
                    self.assertIn('Answer',events)
                    self.assertEqual(len(seen), 3 if combined else 2)
                    if combined:
                        self.assertIn({'sources':['Web — 1 result']},events)

    def test_docs_off_not_offered_to_any_provider(self):
        for provider in llm.PROVIDERS:
            helper = '_claude_turn' if provider == 'Claude' else '_compatible_turn'
            def turn(*args, **kwargs):
                allowed = args[2] if provider == 'Claude' else args[4]
                self.assertNotIn('read_document', allowed)
                yield 'Chat answer'
                return type('Final', (), {'stop_reason':'end_turn'})() if provider == 'Claude' else ('Chat answer', [])
            with patch.object(llm,'get_config',return_value='dummy'), patch.object(llm,helper,side_effect=turn), patch.object(rag,'retrieve') as retrieve:
                list(llm.ask(provider,[{'role':'user','content':'hello'}],False,False,['private document']))
                retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
