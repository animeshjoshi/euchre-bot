import json
import os
import unittest

from euchre import webbridge as wb

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "euchre_q.npz")


def call(fn, *args):
    return json.loads(fn(*args))


class BridgeTest(unittest.TestCase):
    def setUp(self):
        wb._s.update(game=None)
        self.assertTrue(call(wb.init, "/no/such/model.npz")["ok"])

    def test_everything_is_json(self):
        out = call(wb.new_game, json.dumps({"sims": 10, "seed": 1}))
        self.assertTrue(out["ok"])
        self.assertEqual(out["state"]["phase"], "bidding")
        self.assertFalse(out["state"]["has_model"])

    def test_step_and_act_before_a_game_report_errors(self):
        self.assertEqual(call(wb.step), {"ok": False, "error": "no game in progress"})
        self.assertFalse(call(wb.act, "{}")["ok"])

    def test_bad_answer_comes_back_as_error(self):
        wb.new_game(json.dumps({"sims": 10, "seed": 1}))
        r = call(wb.act, json.dumps({"card": 3}))
        self.assertEqual(r, {"ok": False, "error": "no decision pending"})

    def test_a_human_turn_can_be_answered_over_json(self):
        wb.new_game(json.dumps({"sims": 10, "seed": 1, "delay_ms": 0}))
        for _ in range(200):
            out = call(wb.step)
            if out["state"]["pending"]:
                break
        p = out["state"]["pending"]
        self.assertIsNone(out["wait_ms"])
        self.assertTrue(call(wb.act, json.dumps({"kind": "pass", "suit": None, "card": (p.get("legal") or p.get("cards") or [0])[0]}))["ok"])

    def test_analyze_and_its_errors(self):
        ok = call(wb.analyze, json.dumps({"hand": [0, 1, 2, 3, 4], "upcard": 5, "dealer": 3, "round": 1, "sims": 50}))
        self.assertTrue(ok["ok"])
        self.assertGreater(len(ok["analysis"]["results"]), 1)
        bad = call(wb.analyze, json.dumps({"hand": [0, 1], "upcard": 5}))
        self.assertFalse(bad["ok"])

    @unittest.skipUnless(os.path.exists(MODEL), "trained model not present")
    def test_loads_the_model(self):
        self.assertTrue(call(wb.init, MODEL)["has_model"])


if __name__ == "__main__":
    unittest.main()
