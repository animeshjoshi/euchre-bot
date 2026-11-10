import random
import unittest

from euchre.engine import Agent, deal, iter_bidding, run_bidding


class Scripted(Agent):
    """Passes in round 1 and calls a fixed suit in round 2 (or passes if told to)."""

    def __init__(self, order_up=False, call=None, alone=False):
        self.order_up, self.call, self.alone = order_up, call, alone

    def bid_round1(self, view):
        return self.order_up, self.alone

    def bid_round2(self, view):
        return self.call, self.alone

    def discard(self, view, hand6, trump, maker, alone):
        return hand6[0]


class IterBiddingTest(unittest.TestCase):
    def setUp(self):
        self.hands, self.up, _ = deal(random.Random(3))

    def drive(self, agents, dealer=0, stick=False):
        """Drive iter_bidding by hand, the way a UI would."""
        gen = iter_bidding(self.hands, self.up, dealer, stick)
        seen = []
        try:
            req = next(gen)
            while True:
                seen.append(req[0])
                agent = agents[req[1].player]
                if req[0] == "bid1":
                    ans = agent.bid_round1(req[1])
                elif req[0] == "bid2":
                    ans = agent.bid_round2(req[1])
                else:
                    ans = agent.discard(req[1], *req[2:])
                req = gen.send(ans)
        except StopIteration as done:
            return done.value, seen

    def test_everyone_passes(self):
        (contract, _), seen = self.drive([Scripted()] * 4)
        self.assertIsNone(contract)
        self.assertEqual(seen, ["bid1"] * 4 + ["bid2"] * 4)

    def test_order_up_asks_dealer_to_discard(self):
        agents = [Scripted(), Scripted(order_up=True), Scripted(), Scripted()]
        (contract, hands), seen = self.drive(agents, dealer=0)
        self.assertEqual(seen, ["bid1", "discard"])
        self.assertTrue(contract.upcard_taken)
        self.assertEqual(len(hands[0]), 5)
        self.assertNotIn(contract.discard, hands[0])

    def test_dealer_sits_out_when_partner_goes_alone(self):
        # seat 3 deals; seat 1 is the dealer's partner and orders up alone -> dealer sits out
        agents = [Scripted(), Scripted(order_up=True, alone=True), Scripted(), Scripted()]
        (contract, hands), seen = self.drive(agents, dealer=3)
        self.assertEqual(seen, ["bid1", "bid1"])
        self.assertFalse(contract.upcard_taken)
        self.assertEqual(hands[3], self.hands[3])

    def test_run_bidding_matches_manual_drive(self):
        agents = [Scripted(), Scripted(), Scripted(call=(self.up // 6 + 1) % 4), Scripted()]
        manual, _ = self.drive(agents, dealer=1)
        auto = run_bidding(self.hands, self.up, 1, agents)
        self.assertEqual(manual[0], auto[0])
        self.assertEqual(manual[1], auto[1])

    def test_cannot_call_turned_down_suit(self):
        agents = [Scripted(call=self.up // 6)] * 4
        with self.assertRaises(ValueError):
            run_bidding(self.hands, self.up, 0, agents)


if __name__ == "__main__":
    unittest.main()
