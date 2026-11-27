"""Deterministic scripted fixtures, NOT recordings, predictions or device measurements."""
class Replay:
    def __init__(self, scenario):
        self.scenario = scenario
        self.turn = 0
        self.sent = 0
        self.questions = []
        self.histories = []

    def wait_candidate(self): pass
    def rearm(self): pass
    def snapshot(self): return f'모의 센서 상태 {self.turn}'
    def say(self, clip): self.questions.append(clip)
    def listen(self):
        self.turn += 1
        if self.scenario == 'silence': return ''
        if self.scenario == 'clear': return '괜찮아요.'
        return '잘 못 들었어요.' if self.turn == 1 else '도와주세요.'
    def decide(self, history):
        self.histories.append(list(history))
        text = history[-1]['transcript']
        name, reason, route = {'잘 못 들었어요.':('ask_checkin','unclear_reply','repeat_last'),
                               '도와주세요.':('raise_alert','help_requested',None),
                               '괜찮아요.':('close_event','false_alarm',None)}[text]
        return dict(evidence=text,call=dict(name=name,arguments=dict(reason=reason)),next_prompt=route)
    def notify(self, decision):
        self.sent += 1
        return 'simulated', None
