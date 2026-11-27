import unittest
from runtime.replay import Replay
from runtime.controller import monitor, route

class Flow(unittest.TestCase):
    def run_case(self, name, **kwargs):
        io = Replay(name)
        return io, monitor(io,io,one=True,log=lambda _:None,**kwargs)
    def test_help_repeat_and_fresh_observations(self):
        io,out = self.run_case('help', alerts=True)
        self.assertEqual(out,'raise_alert'); self.assertEqual(io.sent,1)
        self.assertEqual(io.questions,['initial_check','initial_check','raise_alert'])
        self.assertNotEqual(io.histories[-1][0]['sensor'], io.histories[-1][1]['sensor'])
    def test_no_alert_by_default(self): self.assertEqual(self.run_case('help')[0].sent,0)
    def test_clear(self):
        io,out=self.run_case('clear',alerts=True);self.assertEqual((out,io.sent),('close_event',0))
    def test_silence(self):
        io,out=self.run_case('silence',alerts=True);self.assertEqual((out,io.sent,len(io.questions)),('unresolved',0,4));self.assertFalse(io.histories)
    def test_no_fabricated_evidence(self):
        io=Replay('help');d=io.decide([{'transcript':'도와주세요.'}]);self.assertIsNone(route(d,'다시요.'));self.assertIsNone(route({},'네'))
