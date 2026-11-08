"""Rule-based agents: a baseline bidder/player, used as the opponent model inside Monte Carlo
rollouts and as the benchmark the RL player is trained and evaluated against."""
import random

from .cards import ACE, EFF, TRUMP_POWER, power, suit_of
from .engine import Agent

_TRUMP_VALUE = {13: 3.0, 12: 2.5, 11: 2.0, 10: 1.6, 9: 1.3, 7: 1.1, 6: 1.0}


def trump_strength(hand, t):
    """Crude hand strength if t were trump (~6.5 is a normal 'make it' hand)."""
    s, n_trump, offsuits = 0.0, 0, set()
    tp = TRUMP_POWER[t]
    for c in hand:
        if tp[c] >= 0:
            n_trump += 1
            s += _TRUMP_VALUE[tp[c]]
        else:
            offsuits.add(suit_of(c))
            if c % 6 == ACE:
                s += 0.8
    if n_trump >= 2:
        s += 0.4 * (3 - len(offsuits))      # voids let you trump in
    return s


def best_discard(hand6, t):
    return max(hand6, key=lambda c: (trump_strength([x for x in hand6 if x != c], t), -(c % 6)))


class HeuristicAgent(Agent):
    def __init__(self, order_threshold=6.5, alone_threshold=10.5, upcard_weight=0.75):
        self.order_t = order_threshold
        self.alone_t = alone_threshold
        self.up_w = upcard_weight

    # ---- bidding
    def bid_round1(self, v):
        t = suit_of(v.upcard)
        up_val = _TRUMP_VALUE[TRUMP_POWER[t][v.upcard]]
        base = trump_strength(v.hand, t)
        if v.player == v.dealer:
            h6 = v.hand + [v.upcard]
            d = best_discard(h6, t)
            s = trump_strength([c for c in h6 if c != d], t)
            alone_s = s
        elif (v.player + 2) % 4 == v.dealer:
            s = base + self.up_w * up_val
            alone_s = base                    # dealer sits out -> no pickup
        else:
            s = base - self.up_w * up_val
            alone_s = s
        if v.position == 0:
            s += 0.3                          # first seat leads
        go = s >= self.order_t
        return go, go and alone_s >= self.alone_t

    def bid_round2(self, v):
        up = suit_of(v.upcard)
        scores = [(trump_strength(v.hand, t) + (0.3 if v.position == 0 else 0), t)
                  for t in range(4) if t != up]
        s, t = max(scores)
        if s >= self.order_t or v.must_call:
            return t, s >= self.alone_t
        return None, False

    def discard(self, v, hand6, trump, maker, alone):
        return best_discard(hand6, trump)

    # ---- card play
    def play(self, state):
        legal = state.legal()
        if len(legal) == 1:
            return legal[0]
        p, t = state.current, state.trump
        tp = TRUMP_POWER[t]

        def low_key(c):  # lowest-value card, keep trumps
            return (tp[c] >= 0, tp[c] if tp[c] >= 0 else c % 6)

        if not state.trick:  # leading
            trumps = [c for c in legal if tp[c] >= 0]
            if state.maker % 2 == p % 2 and trumps:
                hi = max(trumps, key=lambda c: tp[c])
                if tp[hi] == 13 or len(trumps) >= 2:
                    return hi
            aces = [c for c in legal if tp[c] < 0 and c % 6 == ACE]
            if aces:
                return aces[0]
            return min(legal, key=low_key)

        led = EFF[t][state.trick[0][1]]
        wp, wc = state.winning()
        if wp % 2 == p % 2:                      # partner winning: throw low
            return min(legal, key=low_key)
        wpow = power(wc, t, led)
        winners = [c for c in legal if power(c, t, led) > wpow]
        if winners:
            return min(winners, key=lambda c: power(c, t, led))
        return min(legal, key=low_key)


class RandomAgent(HeuristicAgent):
    """Heuristic bidding, uniformly random legal card play."""

    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def play(self, state):
        return self.rng.choice(state.legal())
