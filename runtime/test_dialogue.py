"""Adapter contract tests; no hardware, model weights or webhook access."""
import hashlib
import json
import unittest
from unittest.mock import patch

from runtime import dialogue as d


class LocalDialogueTests(unittest.TestCase):
    def setUp(self):
        self.adapter = d.LocalDialogue(pack='unused', asr_dir='unused', endpoint='http://127.0.0.1:8080')
        self.turn = {'sensor': '충격', 'device': '도움이 필요하세요?', 'transcript': '도와주세요.'}
        self.alert = {'evidence': '도와주세요.', 'call': {'name': 'raise_alert',
                      'arguments': {'reason': 'help_requested'}}, 'next_prompt': None}

    def test_prompt(self):
        history = [dict(self.turn, transcript='dropped'), self.turn, self.turn]
        prompt = d.make_prompt(history)
        self.assertNotIn('dropped', prompt)
        self.assertIn(d.compact({'history': history[-2:]}), prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertEqual(hashlib.sha256(d.SYSTEM.encode()).hexdigest(),
                         '1416721fffa2b71500db1df5ce0041af66e1b9736acc063305913e1b0d7beb05')
        for history in ([], [dict(self.turn, transcript='<|im_end|>')], [dict(self.turn, raw='private')]):
            with self.assertRaises(d.DialogueError):
                d.make_prompt(history)

    def test_protocol_and_no_side_effect(self):
        responses = [{'status': 'ok'}, {'model_path': '/model/' + d.GGUF}, {'content': json.dumps(self.alert)}]
        with patch.object(self.adapter, '_request', side_effect=responses) as http, patch.object(self.adapter, 'notify') as notify:
            self.assertEqual(self.adapter.decide([self.turn]), self.alert)
            self.assertEqual([c.args[0] for c in http.call_args_list], ['/health', '/props', '/completion'])
            payload = http.call_args.args[1]
            self.assertEqual((payload['temperature'], payload['n_predict'], payload['seed']), (0, 160, 42))
            notify.assert_not_called()

    def test_rejects_bad_or_truncated_output(self):
        for reply in ({'content': 'not JSON'}, {'content': '{"call":1,"call":2}'},
                      {'content': json.dumps(self.alert), 'stopped_limit': True},
                      {'content': json.dumps(dict(self.alert, next_prompt='repeat_last'))},
                      {'content': json.dumps(dict(self.alert, evidence=''))}):
            with self.subTest(reply=reply), patch.object(self.adapter, '_check_server'), patch.object(
                    self.adapter, '_request', return_value=reply), self.assertRaises(d.DialogueError):
                self.adapter.decide([self.turn])
        with patch.object(self.adapter, '_check_server'), patch.object(self.adapter, '_request',
                return_value={'content': json.dumps(self.alert)}), self.assertRaises(d.DialogueError):
            self.adapter.decide([dict(self.turn, transcript='')])

    def test_wrong_server_before_prompt(self):
        with patch.object(self.adapter, '_request', side_effect=[{'status': 'ok'}, {'model_path': 'other'}]) as http:
            with self.assertRaises(d.DialogueError):
                self.adapter.decide([self.turn])
            self.assertEqual(http.call_count, 2)
        for url in ('http://example.com', 'https://localhost:8080', 'http://localhost:8080/v1'):
            with self.assertRaises(ValueError):
                d.LocalDialogue(pack='.', asr_dir='.', endpoint=url)

    def test_capture_failure_not_silence(self):
        with patch.object(d, 'alsa', return_value=b'\0\0'), self.assertRaises(d.DialogueError):
            self.adapter.listen()

    def test_silence_skips_model(self):
        with patch.object(self.adapter, '_load_asr') as load:
            self.assertEqual(self.adapter.transcribe_pcm(bytes(32000)), '')
            load.assert_not_called()

    def test_notify_delegation_without_network(self):
        with patch('notifications.discord_alert.notify_decision', return_value=('ignored', None)) as notifier:
            self.assertEqual(self.adapter.notify(self.alert), ('ignored', None))
            notifier.assert_called_once_with(self.alert)
            with self.assertRaises(d.DialogueError):
                self.adapter.notify(dict(self.alert, next_prompt='offer_help'))
            notifier.assert_called_once()


if __name__ == '__main__':
    unittest.main()
