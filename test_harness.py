import copy
import unittest
from harness import rows, validate, score

class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.c = rows('examples/cases.jsonl')
        self.p = rows('examples/mock_predictions.jsonl')
    def test_demo(self):
        s = score(self.c, self.p)
        self.assertEqual((s['exact_correct'], s['missed_help_closed']), (2, 1))
    def test_arguments(self):
        self.p[1]['call']['arguments'] = {}
        s = score(self.c, self.p)
        self.assertEqual((s['name_correct'], s['exact_correct']), (2, 1))
    def test_missing_prediction(self):
        with self.assertRaises(ValueError): score(self.c, self.p[:-1])
    def test_family_leak(self):
        self.c[0]['split'] = 'test'
        with self.assertRaises(ValueError): validate(self.c)
    def test_duplicate_case(self):
        with self.assertRaises(ValueError): validate(self.c + [copy.deepcopy(self.c[0])])
