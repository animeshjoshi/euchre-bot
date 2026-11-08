"""Determinized Monte Carlo search for card play (play-time lookahead).

For each decision: sample K deals of the hidden cards consistent with what the player has
seen (cards played, known voids, dealer holding the picked-up upcard, hand sizes), play each
legal card, finish the hand with a rollout policy (heuristic or RL), and pick the card with
the best average signed points. Same machinery the rollout trainer uses to make targets.
"""
import random

import numpy as np

from .cards import ALL_CARDS, EFF
from .heuristics import HeuristicAgent


def determinize(state, me, rng, known_extra=(), tries=30):
    t = state.trump
    known = set(state.hands[me]) | set(state.played_cards) | {c for _, c in state.trick} | set(known_extra)
    forced = {}
    if (state.upcard_taken and state.dealer != me and state.upcard not in known
            and not state.voids[state.dealer][t]):
        forced[state.dealer] = state.upcard
    known.add(state.upcard)
    pool = [c for c in ALL_CARDS if c not in known]
    others = [p for p in range(4) if p != me and p != state.out]
    need = {p: len(state.hands[p]) - (1 if p in forced else 0) for p in others}
    for attempt in range(tries):
        rng.shuffle(pool)
        avail = list(pool)
        hands = [None] * 4
        hands[me] = list(state.hands[me])
        if state.out >= 0:
            hands[state.out] = []
        ok = True
        use_voids = attempt < tries - 1
        for p in sorted(others, key=lambda q: -sum(state.voids[q])):
            v = state.voids[p]
            elig = [c for c in avail if not (use_voids and v[EFF[t][c]])]
            if len(elig) < need[p]:
                ok = False
                break
            h = elig[:need[p]]
            for c in h:
                avail.remove(c)
            hands[p] = h + ([forced[p]] if p in forced else [])
        if ok:
            return hands
    return None


def rollout_values(state, rollout_policy, k, rng, known_extra=()):
    """{card: mean signed points for the acting seat's team} over k determinizations."""
    me = state.current
    legal = state.legal()
    sums = {c: 0.0 for c in legal}
    n = 0
    for _ in range(k):
        hs = determinize(state, me, rng, known_extra)
        if hs is None:
            continue
        n += 1
        for a in legal:
            s2 = state.clone(hs)
            s2.play(a)
            while not s2.done:
                s2.play(rollout_policy.play(s2))
            pts = s2.points()
            sums[a] += pts[me % 2] - pts[1 - me % 2]
    return {c: v / max(n, 1) for c, v in sums.items()}


class SearchPlayer(HeuristicAgent):
    """Play-time determinized search on top of a rollout policy (default: heuristic)."""

    def __init__(self, rollout_policy=None, k=24, seed=None):
        super().__init__()
        self.rollout = rollout_policy or HeuristicAgent()
        self.k = k
        self.rng = random.Random(seed)

    def q_values(self, state):
        return rollout_values(state, self.rollout, self.k, self.rng)

    def play(self, state):
        legal = state.legal()
        if len(legal) == 1:
            return legal[0]
        qs = self.q_values(state)
        return max(qs, key=qs.get)
