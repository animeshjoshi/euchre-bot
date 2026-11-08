"""Duplicate matches: every deal is played twice with the two teams swapping seats,
which cancels most of the luck of the cards."""
import math
import random

from .engine import deal, play_out, run_bidding


def duplicate_match(bot_a, bot_b, n_deals=1000, seed=0, stick_the_dealer=False):
    """Average signed points per hand for team A (positive = A better). Returns (mean, stderr)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_deals):
        hands, upcard, _ = deal(rng)
        dealer = rng.randrange(4)
        total = 0.0
        for a_team in (0, 1):
            seats = [bot_a if p % 2 == a_team else bot_b for p in range(4)]
            contract, h2 = run_bidding(hands, upcard, dealer, seats, stick_the_dealer)
            if contract is None:
                continue
            pts = play_out(h2, contract, dealer, upcard, seats).points()
            total += pts[a_team] - pts[1 - a_team]
        diffs.append(total / 2)
    m = sum(diffs) / len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
    return m, math.sqrt(var / len(diffs))
