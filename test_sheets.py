import json
from queue import Queue
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from streamlit.testing.v1 import AppTest
import llm
import sheets


INFO = json.dumps({'type':'service_account','client_email':'test@example.com', 'private_key':'private-test-secret', 'token_uri':'https://oauth2.googleapis.com/token'})


class SheetsTests(unittest.TestCase):
    def test_missing_either_setting_skips(self):
        for missing in ('TARGET_SHEET_ID', 'GOOGLE_SERVICE_ACCOUNT_JSON'):
            with patch.object(sheets,'get_config',side_effect=lambda key: '' if key == missing else 'dummy'), patch.object(sheets,'Thread') as thread, patch.object(sheets,'_jobs') as queue:
                sheets.log_conversation('Claude','model','Hi',{'content':'Answer'})
                thread.assert_not_called()
                queue.put_nowait.assert_not_called()

    def test_queue_snapshots_credentials_and_row(self):
        queue = Queue()
        values = {'GOOGLE_SERVICE_ACCOUNT_JSON': INFO, 'TARGET_SHEET_ID':'sheet-one'}
        reply = {'content':'Answer','sources':['Web — 3 results']}
        with patch.object(sheets,'_worker',MagicMock()), patch.object(sheets,'_jobs',queue), patch.object(sheets,'get_config',side_effect=lambda key:values.get(key,'')):
            sheets.log_conversation('OpenAI','gpt-4o','=literal',reply)
            values['TARGET_SHEET_ID']='different-sheet'
            reply['content']='changed'
        job = queue.get_nowait()
        self.assertEqual(job[1], 'sheet-one')
        self.assertEqual(job[2][1:6], ['OpenAI','gpt-4o','=literal','Answer','Web — 3 results'])

    def test_append_payload_auth_and_redacted_failure(self):
        for status in (200, 403):
            with self.subTest(status=status):
                credentials = MagicMock(token='token-secret')
                transport = MagicMock(return_value=SimpleNamespace(status=200,data=b'{"access_token":"token-secret"}'))
                credentials.refresh.side_effect=lambda request: request('https://oauth2.googleapis.com/token',method='POST',body=b'assertion=jwt-secret')
                response=httpx.Response(status,text='{"updates":{"updatedRows":1}}' if status==200 else 'raw denied private-test-secret token-secret jwt-secret',request=httpx.Request('POST','https://sheets.googleapis.com'))
                with patch.object(sheets.service_account.Credentials,'from_service_account_info',return_value=credentials) as factory, patch.object(sheets,'Request',return_value=transport), patch.object(sheets.httpx,'post',return_value=response) as post, patch.object(sheets,'logger') as log:
                    row=['time','OpenAI','gpt-4o','=literal','Answer','None','']
                    sheets._append(INFO,'sheet-one',row,(INFO,'sheet-one'))
                    self.assertEqual(factory.call_args.kwargs['scopes'],['https://www.googleapis.com/auth/spreadsheets'])
                    self.assertEqual(post.call_args.kwargs['json']['values'],[row])
                    self.assertEqual(post.call_args.kwargs['params']['valueInputOption'],'RAW')
                    self.assertEqual(post.call_args.kwargs['params']['insertDataOption'],'INSERT_ROWS')
                    self.assertIn('/sheet-one/values/A%3AG:append',post.call_args.args[0])
                    self.assertEqual(transport.call_args.kwargs['timeout'],30)
                    logged=str(log.mock_calls)
                    for secret in ('private-test-secret','token-secret','jwt-secret'):
                        self.assertNotIn(secret,logged)
                    if status==403:
                        self.assertIn('403',logged)
                        self.assertIn('raw denied',logged)

    def test_bad_json_and_auth_failure_do_not_append(self):
        with patch.object(sheets.httpx,'post') as post, patch.object(sheets,'logger') as log:
            sheets._append('invalid json','sheet',[],('invalid json',))
            post.assert_not_called()
            self.assertIn('JSONDecodeError',str(log.error.call_args))
        credentials=MagicMock()
        credentials.refresh.side_effect=RuntimeError('auth unavailable')
        with patch.object(sheets.service_account.Credentials,'from_service_account_info',return_value=credentials), patch.object(sheets.httpx,'post') as post:
            sheets._append(INFO,'sheet',[],(INFO,))
            post.assert_not_called()

    def test_worker_does_not_block_enqueue(self):
        entered, release = Event(), Event()
        def slow_append(*job):
            entered.set()
            release.wait(5)
        with patch.object(sheets,'get_config',side_effect=lambda key: INFO if key=='GOOGLE_SERVICE_ACCOUNT_JSON' else 'sheet' if key=='TARGET_SHEET_ID' else ''), patch.object(sheets,'_append',side_effect=slow_append):
            try:
                sheets.log_conversation('Claude','model','Hi',{'content':'Answer'})
                self.assertTrue(entered.wait(2))
                self.assertFalse(release.is_set())
            finally:
                release.set()
                sheets._jobs.join()

    def test_ui_enqueues_once_after_reply_not_rerun(self):
        with patch.object(llm,'ask',return_value=iter(['Answer'])), patch.object(sheets,'log_conversation') as log:
            app=AppTest.from_file('app.py').run()
            app.chat_input[0].set_value('Question').run()
            self.assertFalse(app.exception)
            log.assert_called_once()
            self.assertEqual(log.call_args.args[3]['content'],'Answer')
            app.run()
            log.assert_called_once()


if __name__=='__main__':
    unittest.main()
