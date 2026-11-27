"""Discord output for one validated Fall Agent event; standard library only."""
import argparse
import http.client
import json
import os
import re
import sys
from urllib.parse import urlsplit

HELP = {"name": "raise_alert", "arguments": {"reason": "help_requested"}}
TEST_TEXT = "[AIoT Fall Agent] 연결 테스트 — 실제 상황 아님. 도움 요청 알림 경로가 연결되었습니다."
LIVE_TEXT = "[AIoT Fall Agent] 장치에서 도움 요청이 감지되었습니다. 현장 상태를 확인해 주세요."


def notify_decision(decision, *, test=False):
    """Returns (status, message_id). Call once per confirmed event, not per frame."""
    if not isinstance(decision, dict) or decision.get('call') != HELP or 'next_prompt' not in decision or decision['next_prompt'] is not None:
        return 'ignored', None
    url = os.environ.get('FALL_AGENT_DISCORD_WEBHOOK_URL', '')
    if not url:
        return 'not_configured', None
    conn = None
    try:
        u = urlsplit(url)
        if (u.scheme != 'https' or u.netloc not in ('discord.com', 'discordapp.com') or u.query or u.fragment or
                not re.fullmatch(r'/api/(?:v[0-9]+/)?webhooks/[0-9]+/[A-Za-z0-9_-]+', u.path)):
            return 'invalid_config', None
        conn = http.client.HTTPSConnection(u.netloc, timeout=4)
        payload = {'username': 'AIoT Fall Agent', 'content': TEST_TEXT if test else LIVE_TEXT, 'allowed_mentions': {'parse': []}, 'tts': False}
        conn.request('POST', u.path+'?wait=true', json.dumps(payload).encode(), {'Content-Type':'application/json','User-Agent':'AIoT-Fall-Agent/1.0'})
        response = conn.getresponse()
        if response.status != 200:
            return 'http_'+str(response.status), None
        msg = json.loads(response.read(65536))
        if not isinstance(msg, dict) or not str(msg.get('id', '')).isdigit() or msg.get('content') != payload['content']:
            return 'unconfirmed', None
        return 'sent', str(msg['id'])
    except (ValueError, OSError, http.client.HTTPException):
        return 'unconfirmed', None  # never print URLs or retry an ambiguous POST
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--test', action='store_true', help='send one visibly labelled test')
    mode.add_argument('--decision-stdin', action='store_true', help='consume one model decision as JSON')
    args = parser.parse_args()
    try:
        decision = {'call': HELP, 'next_prompt': None} if args.test else json.load(sys.stdin)
    except ValueError:
        parser.exit(2, 'invalid decision JSON\n')
    status, mid = notify_decision(decision, test=args.test)
    print(json.dumps({'status': status, 'message_id': mid}))
    sys.exit(0 if status in ('sent','ignored') else 1)
