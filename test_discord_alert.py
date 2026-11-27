import os
import unittest
from unittest.mock import patch, MagicMock
from notifications import discord_alert as api


class Alerts(unittest.TestCase):
    @patch('notifications.discord_alert.http.client.HTTPSConnection')
    def test_gate(self, conn):
        for obj in (None, {}, {'call':api.HELP}, {'call':api.HELP,'next_prompt':'offer_help'}, {'call':{'name':'close_event'},'next_prompt':None}, {'call':{'name':'raise_alert','arguments':{'reason':'unclear_reply'}},'next_prompt':None}):
            self.assertEqual(api.notify_decision(obj), ('ignored', None))
        conn.assert_not_called()

    @patch.dict(os.environ, {'FALL_AGENT_DISCORD_WEBHOOK_URL':''})
    def test_missing(self):
        self.assertEqual(api.notify_decision({'call':api.HELP,'next_prompt':None})[0], 'not_configured')

    @patch.dict(os.environ, {'FALL_AGENT_DISCORD_WEBHOOK_URL':'https://discord.com/api/webhooks/123/example'})
    @patch('notifications.discord_alert.http.client.HTTPSConnection')
    def test_transport(self, conn):
        import json
        c = conn.return_value; response = c.getresponse.return_value; response.status = 200
        d = {'call':api.HELP,'next_prompt':None,'evidence':'PRIVATE'}
        for test, text in ((True,api.TEST_TEXT),(False,api.LIVE_TEXT)):
            response.read.return_value=json.dumps({'id':'123','content':text}).encode()
            self.assertEqual(api.notify_decision(d,test=test),('sent','123'))
            payload=json.loads(c.request.call_args.args[2]); self.assertEqual(payload['content'],text)
            self.assertEqual(payload['allowed_mentions'],{'parse':[]});self.assertNotIn('PRIVATE',str(payload))
        c.request.side_effect=TimeoutError(); self.assertEqual(api.notify_decision(d),('unconfirmed',None)); self.assertEqual(c.request.call_count,3)
