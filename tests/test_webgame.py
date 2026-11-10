import os
import random
import unittest

from euchre.heuristics import HeuristicAgent
from euchre.webgame import Game, analyze_bid, load_player

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "euchre_q.npz")


def human_answer(p, rng):
    """A simple scripted human: bids now and then, otherwise plays the first legal card."""
    t = p["type"]
    if t == "bid1":
        return {"kind": "order"} if rng.random() < 0.3 else {"kind": "pass"}
    if t == "bid2":
        return {"suit": p["suits"][0]} if p["must_call"] or rng.random() < 0.4 else {"suit": None}
    if t == "discard":
        return {"card": p["cards"][0]}
    if t == "play":
        return {"card": p["legal"][0]}
    return {}


def play_game(game, rng, max_steps=20000):
    """Drive a game to the end the way the page does. Returns every state seen."""
    seen = []
    for _ in range(max_steps):
        out = game.step()
        st = out["state"]
        seen.append((st["phase"], out["wait_ms"], st["pending"] and st["pending"]["type"]))
        p = st["pending"]
        if out["wait_ms"] is None and p is None:
            return seen, st          # nothing pending and nothing to wait for: game finished
        if p is not None:
            if p["type"] == "new_game":
                return seen, st
            assert out["wait_ms"] is None
            assert game.submit(human_answer(p, rng)) is None
    raise AssertionError("game did not finish")


class WebGameTest(unittest.TestCase):
    def new_game(self, **kw):
        return Game(HeuristicAgent(), False, sims=10, hint_sims=10, delay_ms=500, target=4, seed=5, **kw)

    def test_plays_a_whole_game(self):
        game = self.new_game()
        seen, final = play_game(game, random.Random(1))
        self.assertEqual(final["phase"], "game_over")
        self.assertGreaterEqual(max(final["score"]), 4)
        self.assertIn(final["result"], ("You win the game!", "The bots win the game."))
        # the page gets to show "thinking" and the bots' moves with a delay
        self.assertTrue(any(w == 500 for _, w, _ in seen))
        self.assertTrue(any(w == 0 for _, w, _ in seen))

    def test_same_seed_and_answers_give_the_same_game(self):
        a = play_game(self.new_game(), random.Random(9))[1]
        b = play_game(self.new_game(), random.Random(9))[1]
        self.assertEqual((a["score"], a["log"], a["hand_no"]), (b["score"], b["log"], b["hand_no"]))

    def test_step_does_not_advance_while_waiting_for_the_human(self):
        game = self.new_game()
        for _ in range(500):
            out = game.step()
            if out["state"]["pending"]:
                break
        first = repr(out["state"])
        for _ in range(3):
            again = game.step()
            self.assertIsNone(again["wait_ms"])
        self.assertEqual(repr(again["state"]), first)

    def test_rejects_bad_answers(self):
        game = self.new_game()
        self.assertEqual(game.submit({"card": 1}), "no decision pending")
        rng = random.Random(2)
        while True:
            out = game.step()
            p = out["state"]["pending"]
            if p and p["type"] == "play":
                break
            if p:
                game.submit(human_answer(p, rng))
        illegal = next(c for c in range(24) if c not in p["legal"])
        self.assertEqual(game.submit({"card": illegal}), "illegal card")
        self.assertIsNone(game.submit({"card": p["legal"][0]}))
        self.assertEqual(game.submit({"card": p["legal"][0]}), "no decision pending")  # only one answer per question

    def run_until(self, game, kind, seed=6):
        """Step the game (bidding pass, other prompts answered) until the human faces a `kind` prompt."""
        rng = random.Random(seed)
        for _ in range(5000):
            p = game.step()["state"]["pending"]
            if p and p["type"] == kind:
                return p
            if p:
                game.submit({"kind": "pass"} if p["type"] == "bid1" else human_answer(p, rng))
        self.fail(f"never reached a {kind} prompt")

    def test_malformed_answers_are_refused_and_the_question_stays_open(self):
        game = self.new_game()
        self.run_until(game, "bid1")
        for bad in (None, 5, "pass", [], {}, {"kind": "maybe"}):
            self.assertIsNotNone(game.submit(bad), bad)
        self.assertIsNone(game.step()["wait_ms"])  # still waiting for the human
        self.assertIsNone(game.submit({"kind": "pass"}))

    def test_bid2_rejects_the_turned_down_suit(self):
        game = self.new_game()
        for bot in game.bots.values():
            bot.bid_round1 = lambda view: (False, False)  # everyone passes round 1, so round 2 happens
        p = self.run_until(game, "bid2")
        up = game.v["upcard"] // 6
        self.assertNotIn(up, p["suits"])
        self.assertEqual(game.submit({"suit": up}), "not a legal suit")
        self.assertIsNone(game.submit({"suit": p["suits"][0]}))

    def test_a_crash_inside_the_game_is_raised_not_hidden(self):
        game = self.new_game()
        game._steps = iter(lambda: 1 / 0, None)
        with self.assertRaises(ZeroDivisionError):
            game.step()
        self.assertEqual(game.step()["wait_ms"], None)  # and the dead game stays dead

    def test_stick_the_dealer_never_redeals_in_round_two(self):
        game = self.new_game(stick=True)
        _, final = play_game(game, random.Random(4))
        self.assertEqual(final["phase"], "game_over")
        self.assertFalse(any("Everyone passed" in line for line in final["log"]))

    @unittest.skipUnless(os.path.exists(MODEL), "trained model not present")
    def test_plays_with_the_trained_model(self):
        player, has_model = load_player(MODEL)
        self.assertTrue(has_model)
        game = Game(player, True, sims=10, hint_sims=10, target=2, seed=3)
        _, final = play_game(game, random.Random(1))
        self.assertEqual(final["phase"], "game_over")

    def test_missing_model_falls_back_to_heuristic(self):
        player, has_model = load_player("/no/such/model.npz")
        self.assertFalse(has_model)
        self.assertIsInstance(player, HeuristicAgent)


class AnalyzeBidTest(unittest.TestCase):
    def test_returns_sorted_options(self):
        out = analyze_bid({"hand": [0, 1, 2, 3, 4], "upcard": 5, "dealer": 3, "round": 1, "sims": 50},
                          HeuristicAgent(), False)
        evs = [r["ev"] for r in out["results"]]
        self.assertEqual(evs, sorted(evs, reverse=True))
        self.assertEqual(out["rollout"], "heuristic")

    def test_rejects_bad_hands(self):
        with self.assertRaises(ValueError):
            analyze_bid({"hand": [0, 1, 2, 3], "upcard": 5}, HeuristicAgent(), False)
        with self.assertRaises(ValueError):
            analyze_bid({"hand": [0, 1, 2, 3, 4], "upcard": 4}, HeuristicAgent(), False)


if __name__ == "__main__":
    unittest.main()
