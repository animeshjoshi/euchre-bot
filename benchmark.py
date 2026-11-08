"""Duplicate-match benchmarks (positive = first bot wins that many points per hand).

    python benchmark.py --deals 2000 --bid-deals 300
"""
import argparse

from euchre.engine import Bot
from euchre.evaluate import duplicate_match
from euchre.heuristics import HeuristicAgent, RandomAgent
from euchre.mc_bidder import MonteCarloBidder
from euchre.rl import RLPlayer
from euchre.search import SearchPlayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/euchre_q.npz")
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--bid-deals", type=int, default=300)
    ap.add_argument("--sims", type=int, default=200)
    a = ap.parse_args()

    h = HeuristicAgent()
    rl = RLPlayer(a.model)
    rows = [
        ("Card play: RL vs heuristic", Bot(h, rl), Bot(h, h), a.deals),
        ("Card play: RL vs random", Bot(h, rl), Bot(h, RandomAgent(1)), a.deals),
        ("Card play: heuristic vs random", Bot(h, h), Bot(h, RandomAgent(1)), a.deals),
        ("Card play: RL + search (k=16) vs heuristic", Bot(h, SearchPlayer(rl, k=16, seed=3)), Bot(h, h), a.bid_deals),
        ("Bidding: MC vs heuristic (both RL play)",
         Bot(MonteCarloBidder(a.sims, play_policy=rl, seed=1), rl), Bot(h, rl), a.bid_deals),
        ("Full bot (MC+RL) vs heuristic bot",
         Bot(MonteCarloBidder(a.sims, play_policy=rl, seed=2), rl), Bot(h, h), a.bid_deals),
    ]
    for name, x, y, n in rows:
        m, se = duplicate_match(x, y, n, seed=999)
        print(f"{name:<42} {m:+.3f} +/- {se:.3f} pts/hand  ({n} duplicate deals)", flush=True)


if __name__ == "__main__":
    main()
