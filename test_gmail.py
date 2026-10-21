import unittest
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import anthropic
import gmail
import llm


def message_bytes(html=False):
    message = EmailMessage()
    message['Subject'] = 'Budget café'
    message['From'] = 'sender@example.com'
    if html:
        message.set_content('<p>Budget 42</p><script>hidden</script>', subtype='html')
    else:
        message.set_content('Budget 42 ' + 'x' * 1500)
        message.add_attachment(b'attachment content', maintype='application', subtype='octet-stream', filename='a.bin')
    return message.as_bytes()


class GmailTests(unittest.TestCase):
    def setUp(self):
        self.mail = MagicMock()
        self.mail.welcome = b'OK ready'
        self.mail.login.return_value = ('OK', [b'authenticated'])
        self.mail.select.return_value = ('OK', [b'3'])
        self.mail.fetch.return_value = ('OK', [(b'1 (BODY[] {123})', message_bytes()), b')'])
        self.mail.logout.return_value = ('BYE', [b'goodbye'])

    def fetch(self):
        def config(key):
            return {'GMAIL_USER': 'test@example.com', 'GMAIL_APP_PASSWORD': 'abcd efgh\u00a0ijkl mnop'}.get(key, '')
        def connect(*args, **kwargs):
            self.assertEqual(gmail.imaplib._MAXLINE, 10000000)
            self.assertEqual(args, ('imap.gmail.com', 993))
            self.assertTrue(kwargs['ssl_context'].check_hostname)
            return self.mail
        with patch.object(gmail, 'get_config', side_effect=config), patch.object(gmail.imaplib, 'IMAP4_SSL', side_effect=connect), patch.object(gmail, 'logger') as log:
            result = gmail.fetch_recent()
            self.assertNotIn('abcdefghijklmnop', str(log.mock_calls))
            self.assertNotIn('abcd efgh', str(log.mock_calls))
        self.mail.search.assert_not_called()
        self.mail.store.assert_not_called()
        self.mail.close.assert_not_called()
        self.mail.logout.assert_called_once()
        return result, log

    def test_small_inbox_readonly_password_and_body_limit(self):
        result, log = self.fetch()
        self.mail.login.assert_called_once_with('test@example.com', 'abcdefghijklmnop')
        self.mail.sock.settimeout.assert_called_once_with(30)
        self.mail.select.assert_called_once_with('INBOX', readonly=True)
        self.mail.fetch.assert_called_once_with('1:3', '(BODY.PEEK[])')
        self.assertEqual(len(result['results'][0]['body']), 1000)
        self.assertEqual(result['results'][0]['subject'], 'Budget café')
        self.assertIn('authenticated', str(log.mock_calls))
        self.assertIn('BODY[]', str(log.mock_calls))

    def test_large_inbox_and_empty_fallback(self):
        self.mail.select.return_value = ('OK', [b'1000000'])
        self.fetch()
        self.mail.fetch.assert_called_once_with('999991:1000000', '(BODY.PEEK[])')
        self.setUp()
        self.mail.select.side_effect = [('OK', [b'0']), ('OK', [b'2'])]
        self.fetch()
        self.assertEqual(self.mail.select.call_args.args, ('"[Gmail]/All Mail"',))
        self.assertTrue(all(c.kwargs['readonly'] for c in self.mail.select.call_args_list))
        self.mail.fetch.assert_called_once_with('1:2', '(BODY.PEEK[])')

    def test_both_mailboxes_empty(self):
        self.mail.select.return_value = ('OK', [b'0'])
        result, _ = self.fetch()
        self.assertEqual(result, {'results': []})
        self.mail.fetch.assert_not_called()

    def test_login_exception_and_select_fetch_errors_are_clean(self):
        for stage in ('login', 'select', 'fetch'):
            with self.subTest(stage=stage):
                self.setUp()
                getattr(self.mail, stage).side_effect = RuntimeError('Detailed server rejection')
                result, log = self.fetch()
                self.assertIn('⚠️ Gmail Diagnostic: Detailed server rejection', result['error'])
                self.assertIn('RuntimeError', str(log.error.call_args))
                self.assertNotIn('Traceback', result['error'])
        self.setUp()
        self.mail.login.return_value = ('NO', [b'rejected'])
        self.assertIn('Diagnostic', self.fetch()[0]['error'])

    def test_missing_credentials_no_connection(self):
        with patch.object(gmail, 'get_config', return_value=''), patch.object(gmail.imaplib, 'IMAP4_SSL') as connect:
            self.assertIn('unavailable', gmail.fetch_recent()['error'])
            connect.assert_not_called()

    def test_html_and_no_attachment_body(self):
        self.assertEqual(gmail._message(message_bytes(True))['body'], 'Budget 42')
        self.assertNotIn('attachment content', gmail._message(message_bytes())['body'])

    def test_status_reload_order_and_off_gate(self):
        with patch.object(llm.importlib, 'reload') as reload, patch.object(gmail, 'fetch_recent', return_value={'results': []}) as fetch:
            stream = llm._execute('read_email', {'query': 'recent mail'}, True)
            self.assertEqual(next(stream), {'status': '✉️ Checking email…'})
            reload.assert_not_called()
            fetch.assert_not_called()
            self.assertIn({'sources': ['Email — no recent messages']}, list(stream))
            reload.assert_called_once_with(gmail)
            fetch.assert_called_once()
            reload.reset_mock()
            fetch.reset_mock()
            list(llm._execute('read_email', {}, False))
            reload.assert_not_called()
            fetch.assert_not_called()

    def test_all_providers_email_tool_context_and_gate(self):
        for provider in llm.PROVIDERS:
            for enabled in (True, False):
                with self.subTest(provider=provider, enabled=enabled):
                    def turn(*args, **kwargs):
                        history, allowed = (args[1], args[2]) if provider == 'Claude' else (args[3], args[4])
                        if 'read_email' in allowed:
                            self.assertTrue(enabled)
                            if provider == 'Claude':
                                return anthropic.types.Message(id='msg', type='message', role='assistant', model=llm.MODEL,
                                    content=[{'type':'tool_use','id':'email1','name':'read_email','input':{'query':'email'}}],
                                    stop_reason='tool_use', usage={'input_tokens':1,'output_tokens':1})
                            return '', [{'id':'email1','type':'function','function':{'name':'read_email','arguments':'{"query":"email"}'}}]
                        if enabled:
                            self.assertIn('Budget 42', str(history))
                        yield 'Answer'
                        return type('Final', (), {'stop_reason':'end_turn'})() if provider == 'Claude' else ('Answer', [])
                    helper = '_claude_turn' if provider == 'Claude' else '_compatible_turn'
                    with patch.object(llm,'get_config',return_value='dummy'), patch.object(llm,helper,side_effect=turn), patch.object(llm.importlib,'reload'), patch.object(gmail,'fetch_recent',return_value={'results':[{'body':'Budget 42'}]}) as fetch:
                        events=list(llm.ask(provider,[{'role':'user','content':'email'}],email_enabled=enabled))
                        self.assertEqual(fetch.call_count,int(enabled))
                        self.assertIn('Answer',events)
                        if enabled:
                            self.assertIn({'sources':['Email — 1 message']},events)


if __name__ == '__main__':
    unittest.main()
