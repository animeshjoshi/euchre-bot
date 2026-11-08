"""Monte Carlo trump-call / pass decision.

The bidder only knows its own 5 cards, the upcard, the dealer seat and that everyone
before it passed. For each decision it:
  1. samples N "worlds": random deals of the 18 unseen cards to the other three hands
     (+3 buried kitty cards), optionally rejecting worlds in which an earlier player's
     modelled bidding would NOT have passed (Bayesian inference from passes);
  2. for every candidate action (pass / order up / order up alone / name suit X [alone],
     and for the dealer each possible discard) plays the hand to the end in that world
     with a rollout play policy (heuristic or the trained RL player);
     "pass" is valued by continuing the bidding with the opponent model;
  3. uses the same worlds for every action (common random numbers -> low-variance
     comparisons) and picks the action with the highest expected signed points
     (+points my team scores, -points the opponents score).
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .cards import ALL_CARDS, SUITS, card_str, suit_of
from .engine import Agent, BidView, Contract, apply_order_up, play_out, run_bidding
from .heuristics import HeuristicAgent


@dataclass
class Option:
    kind: str                 # 'pass' | 'order' | 'call' | 'discard'
    suit: Optional[int] = None
    alone: bool = False
    discard: Optional[int] = None

    def label(self):
        if self.kind == "pass":
            return "pass"
        a = " ALONE" if self.alone else ""
        if self.kind == "order":
            d = f" (discard {card_str(self.discard)})" if self.discard is not None else ""
            return f"order up {SUITS[self.suit]}{a}{d}"
        if self.kind == "call":
            return f"call {SUITS[self.suit]}{a}"
        return f"discard {card_str(self.discard)}"


@dataclass
class OptionResult:
    option: Option
    ev: float
    se: float
    n: int
    p_make: float = field(default=float("nan"))  # P(my team scores > 0)

    def __str__(self):
        return f"{self.option.label():<32} EV {self.ev:+.3f} +/- {self.se:.3f}   P(win hand) {self.p_make:5.1%}"


class MonteCarloBidder(Agent):
    def __init__(self, n_sims=400, play_policy=None, opponent_model=None,
                 condition_on_passes=True, stick_the_dealer=False, consider_alone=True,
                 max_reject_tries=30, seed=None):
        self.n_sims = n_sims
        self.play_policy = play_policy or HeuristicAgent()
        self.model = opponent_model or HeuristicAgent()
        self.condition = condition_on_passes
        self.stick = stick_the_dealer
        self.consider_alone = consider_alone
        self.max_tries = max_reject_tries
        self.rng = random.Random(seed)
        self.last_analysis = None

    # ------------------------------------------------------------ options
    def _options(self, v: BidView):
        alones = (False, True) if self.consider_alone else (False,)
        up = suit_of(v.upcard)
        opts = []
        if v.round == 1:
            opts.append(Option("pass"))
            for a in alones:
                if v.player == v.dealer:
                    for d in v.hand + [v.upcard]:
                        opts.append(Option("order", up, a, d))
                else:
                    opts.append(Option("order", up, a))
        else:
            if not v.must_call:
                opts.append(Option("pass"))
            for t in range(4):
                if t != up:
                    for a in alones:
                        opts.append(Option("call", t, a))
        return opts

    # ----------------------------------------------------------- sampling
    def _sample_world(self, v: BidView, known):
        unknown = [c for c in ALL_CARDS if c not in known]
        others = [(v.player + i) % 4 for i in (1, 2, 3)]
        hands = None
        for _ in range(self.max_tries):
            self.rng.shuffle(unknown)
            hands = [None] * 4
            hands[v.player] = list(v.hand)
            for i, q in enumerate(others):
                hands[q] = unknown[i * 5:(i + 1) * 5]
            if not self.condition or self._consistent(v, hands):
                break
        return hands

    def _consistent(self, v: BidView, hands):
        """Would the modelled players before me have passed with these hands?"""
        order = v.order
        r1_passers = order if v.round == 2 else order[:v.position]
        for i, q in enumerate(r1_passers):
            if q == v.player:
                continue
            if self.model.bid_round1(BidView(q, hands[q], v.upcard, v.dealer, 1, i))[0]:
                return False
        if v.round == 2:
            for i, q in enumerate(order[:v.position]):
                if self.model.bid_round2(BidView(q, hands[q], v.upcard, v.dealer, 2, i))[0] is not None:
                    return False
        return True

    # --------------------------------------------------------- evaluation
    def _signed(self, state, team):
        pts = state.points()
        return pts[team] - pts[1 - team]

    def _value(self, v: BidView, hands, opt: Option):
        p, d, up = v.player, v.dealer, v.upcard
        players = [self.play_policy] * 4
        if opt.kind == "pass":
            nxt_round, nxt_idx = (v.round, v.position + 1)
            contract, h2 = run_bidding(hands, up, d, [self.model] * 4, self.stick, nxt_round, nxt_idx)
            if contract is None:
                return 0.0           # thrown in, redeal
        elif opt.kind == "order":
            t = suit_of(up)

            def disc(h6):
                if d == p:
                    return opt.discard
                dv = BidView(d, hands[d], up, d, 1, v.order.index(d))
                return self.model.discard(dv, h6, t, p, opt.alone)
            h2, taken, dc = apply_order_up(hands, up, d, p, opt.alone, disc)
            contract = Contract(t, p, opt.alone, 1, taken, dc)
        else:  # call
            contract, h2 = Contract(opt.suit, p, opt.alone, 2, False), hands
        state = play_out(h2, contract, d, up, players)
        return self._signed(state, p % 2)

    def analyze(self, v: BidView, n_sims=None):
        """Return OptionResults for every legal bid, best first."""
        n = n_sims or self.n_sims
        opts = self._options(v)
        known = set(v.hand) | {v.upcard}
        s = [0.0] * len(opts)
        s2 = [0.0] * len(opts)
        wins = [0] * len(opts)
        for _ in range(n):
            hands = self._sample_world(v, known)
            for i, o in enumerate(opts):
                x = self._value(v, hands, o)
                s[i] += x
                s2[i] += x * x
                wins[i] += x > 0
        res = []
        for i, o in enumerate(opts):
            m = s[i] / n
            var = max(s2[i] / n - m * m, 0.0)
            res.append(OptionResult(o, m, math.sqrt(var / n), n, wins[i] / n))
        res.sort(key=lambda r: -r.ev)
        self.last_analysis = (v, res)
        return res

    # ----------------------------------------------------- Agent interface
    def bid_round1(self, v):
        best = self.analyze(v)[0].option
        return best.kind == "order", best.alone

    def bid_round2(self, v):
        best = self.analyze(v)[0].option
        return (best.suit, best.alone) if best.kind == "call" else (None, False)

    def analyze_discard(self, v, hand6, trump, maker, alone, n_sims=None):
        """Dealer picked up because someone else ordered: choose discard by MC."""
        n = n_sims or self.n_sims
        p, up = v.player, v.upcard
        known = set(hand6)
        unknown = [c for c in ALL_CARDS if c not in known]
        others = [(p + i) % 4 for i in (1, 2, 3)]
        contract = Contract(trump, maker, alone, 1, True)
        sums = {c: [0.0, 0.0] for c in hand6}
        for _ in range(n):
            self.rng.shuffle(unknown)
            hands = [None] * 4
            for i, q in enumerate(others):
                hands[q] = unknown[i * 5:(i + 1) * 5]
            for c in hand6:
                hands[p] = [x for x in hand6 if x != c]
                x = self._signed(play_out(hands, contract, v.dealer, up, [self.play_policy] * 4), p % 2)
                sums[c][0] += x
                sums[c][1] += x * x
        res = []
        for c, (a, b) in sums.items():
            m = a / n
            res.append(OptionResult(Option("discard", discard=c), m,
                                    math.sqrt(max(b / n - m * m, 0) / n), n))
        res.sort(key=lambda r: -r.ev)
        return res

    def discard(self, v, hand6, trump, maker, alone):
        if maker == v.player:
            # we evaluated discards while deciding to order; reuse that choice
            if self.last_analysis and self.last_analysis[0].player == v.player:
                o = self.last_analysis[1][0].option
                if o.kind == "order" and o.discard in hand6:
                    return o.discard
        return self.analyze_discard(v, hand6, trump, maker, alone, max(50, self.n_sims // 3))[0].option.discard
