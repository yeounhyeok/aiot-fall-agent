"""Offline, one-turn function-call evaluation. No hardware/network calls."""
import argparse
import json
from pathlib import Path

ACTIONS = {"raise_alert": {"help_requested"}, "ask_checkin": {"unclear_reply"}, "close_event": {"false_alarm"}}

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def validate(cases):
    if not cases:
        raise ValueError("empty cases")
    seen, groups = set(), {}
    for c in cases:
        for k in ("id", "family", "split", "sensor", "question", "reply", "source"):
            if not isinstance(c.get(k), str) or not c[k].strip():
                raise ValueError(f"missing/empty {k}: {c.get('id')}")
        if c['id'] in seen or c['split'] not in {'demo','train','val','test'}:
            raise ValueError(f"duplicate ID or unknown split: {c['id']}")
        seen.add(c['id'])
        if groups.setdefault(c['family'], c['split']) != c['split']:
            raise ValueError(f"family crosses splits: {c['family']}")
        call = c['gold']
        if not isinstance(call, dict) or set(call) != {'name','arguments'} or call['name'] not in ACTIONS:
            raise ValueError(f"invalid function: {c['id']}")
        args = call['arguments']
        if not isinstance(args, dict) or set(args) != {'reason'} or args['reason'] not in ACTIONS[call['name']]:
            raise ValueError(f"invalid arguments: {c['id']}")

def score(cases, predictions):
    validate(cases)
    pred = {p['id']: p.get('call') for p in predictions}
    if len(pred) != len(predictions) or set(pred) != {c['id'] for c in cases}:
        raise ValueError('prediction IDs must match the case IDs exactly, without duplicates')
    names = exact = unsafe_close = 0
    for c in cases:
        call, gold = pred[c['id']], c['gold']
        name = call.get('name') if isinstance(call, dict) else None
        names += name == gold['name']
        exact += call == gold
        unsafe_close += name == 'close_event' and gold['name'] == 'raise_alert'
    return dict(total=len(cases), name_correct=names, exact_correct=exact, exact_rate=exact/len(cases), missed_help_closed=unsafe_close)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases'); parser.add_argument('predictions', nargs='?')
    a = parser.parse_args()
    try:
        cases = rows(a.cases); validate(cases)
        print(json.dumps(score(cases, rows(a.predictions)) if a.predictions else {'valid': len(cases)}, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as e:
        parser.exit(2, f'error: {e}\n')
