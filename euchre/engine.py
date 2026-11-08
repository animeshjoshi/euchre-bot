"""Euchre rules engine: dealing, two-round bidding, dealer pick-up, trick play, scoring.

Seats 0..3 clockwise; teams are seat % 2 (0&2 vs 1&3).
Scoring per hand (makers = team that named trump):
    3-4 tricks -> 1 point, 5 tricks -> 2 points (4 if alone), <3 tricks -> defenders get 2 (euchre).
"""
import random
from dataclasses import dataclass
from typing import Optional

from .cards import EFF, NUM_CARDS, power, suit_of


# ---------------------------------------------------------------- agents --
class Agent:
    """Interface every bot implements. Bidding methods receive a BidView (own info only)."""

    def bid_round1(self, view) -> tuple:  # -> (order_up: bool, alone: bool)
        raise NotImplementedError

    def bid_round2(self, view) -> tuple:  # -> (suit or None, alone: bool)
        raise NotImplementedError

    def discard(self, view, hand6, trump, maker, alone) -> int:
        raise NotImplementedError

    def play(self, state) -> int:  # state: PlayState, acting seat = state.current
        raise NotImplementedError


class Bot(Agent):
    """Combine a bidding agent and a card-play agent into one seat."""

    def __init__(self, bidder, player, name=None):
        self.bidder, self.player = bidder, player
        self.name = name or f"{type(bidder).__name__}+{type(player).__name__}"

    def bid_round1(self, view):
        return self.bidder.bid_round1(view)

    def bid_round2(self, view):
        return self.bidder.bid_round2(view)

    def discard(self, view, hand6, trump, maker, alone):
        return self.bidder.discard(view, hand6, trump, maker, alone)

    def play(self, state):
        return self.player.play(state)


# --------------------------------------------------------------- bidding --
@dataclass
class BidView:
    """Everything a player legitimately knows while bidding."""
    player: int
    hand: list
    upcard: int
    dealer: int
    round: int          # 1 = order up the upcard, 2 = name another suit
    position: int       # 0..3 in bidding order (everyone before has passed)
    must_call: bool = False   # stick-the-dealer in round 2

    @property
    def order(self):
        return [(self.dealer + 1 + i) % 4 for i in range(4)]


@dataclass
class Contract:
    trump: int
    maker: int
    alone: bool
    round: int
    upcard_taken: bool
    discard: Optional[int] = None


def deal(rng=random):
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    hands = [deck[i * 5:(i + 1) * 5] for i in range(4)]
    kitty = deck[20:]
    return hands, kitty[0], kitty


def apply_order_up(hands, upcard, dealer, maker, alone, discard_fn):
    """Dealer picks up the upcard (unless dealer sits out) and discards. Returns (hands, taken, discard)."""
    hands = [list(h) for h in hands]
    out = (maker + 2) % 4 if alone else -1
    if dealer == out:
        return hands, False, None
    hand6 = hands[dealer] + [upcard]
    d = discard_fn(hand6)
    hand6.remove(d)
    hands[dealer] = hand6
    return hands, True, d


def run_bidding(hands, upcard, dealer, bidders, stick_the_dealer=False,
                start_round=1, start_idx=0):
    """Run (the rest of) the bidding. Returns (Contract or None, hands_after_pickup)."""
    order = [(dealer + 1 + i) % 4 for i in range(4)]
    up_suit = suit_of(upcard)
    if start_round == 1:
        for i in range(start_idx, 4):
            p = order[i]
            v = BidView(p, hands[p], upcard, dealer, 1, i)
            go, alone = bidders[p].bid_round1(v)
            if go:
                dv = BidView(dealer, hands[dealer], upcard, dealer, 1, order.index(dealer))
                new_hands, taken, d = apply_order_up(
                    hands, upcard, dealer, p, alone,
                    lambda h6: bidders[dealer].discard(dv, h6, up_suit, p, alone))
                return Contract(up_suit, p, alone, 1, taken, d), new_hands
        start_idx = 0
    for i in range(start_idx, 4):
        p = order[i]
        must = stick_the_dealer and p == dealer
        v = BidView(p, hands[p], upcard, dealer, 2, i, must)
        suit, alone = bidders[p].bid_round2(v)
        if suit is not None:
            if suit == up_suit:
                raise ValueError("cannot name the turned-down suit")
            return Contract(suit, p, alone, 2, False), [list(h) for h in hands]
    return None, hands


# ------------------------------------------------------------------ play --
class PlayState:
    """State of the trick-taking phase. Agents should only look at
    hands[current], trick, history/played_cards, voids, and public contract info."""

    def __init__(self, hands, contract: Contract, dealer, upcard):
        self.hands = [list(h) for h in hands]
        self.trump = contract.trump
        self.maker = contract.maker
        self.alone = contract.alone
        self.dealer = dealer
        self.upcard = upcard
        self.upcard_taken = contract.upcard_taken
        self.out = (self.maker + 2) % 4 if self.alone else -1
        self.n_players = 3 if self.alone else 4
        self.current = self._next(dealer)
        self.trick = []            # [(player, card)] in play order
        self.history = []          # completed tricks: [(player, card)]*n
        self.played_cards = []     # cards from completed tricks
        self.tricks_won = [0, 0]
        self.voids = [[False] * 4 for _ in range(4)]   # voids[player][effective suit]
        self.trick_no = 0

    def clone(self, hands=None):
        """Copy of this state, optionally with the hidden hands replaced (determinization)."""
        s = PlayState.__new__(PlayState)
        s.__dict__.update(self.__dict__)
        s.hands = [list(h) for h in (hands if hands is not None else self.hands)]
        s.trick = list(self.trick)
        s.history = list(self.history)
        s.played_cards = list(self.played_cards)
        s.tricks_won = list(self.tricks_won)
        s.voids = [list(v) for v in self.voids]
        return s

    def _next(self, p):
        p = (p + 1) % 4
        return (p + 1) % 4 if p == self.out else p

    @property
    def done(self):
        return self.trick_no == 5

    def led_suit(self):
        return EFF[self.trump][self.trick[0][1]] if self.trick else None

    def legal(self):
        hand = self.hands[self.current]
        if not self.trick:
            return list(hand)
        e = EFF[self.trump]
        led = e[self.trick[0][1]]
        follow = [c for c in hand if e[c] == led]
        return follow if follow else list(hand)

    def winning(self):
        """(player, card) currently winning the trick in progress."""
        led = self.led_suit()
        t = self.trump
        return max(self.trick, key=lambda pc: power(pc[1], t, led))

    def play(self, card):
        p = self.current
        if card not in self.legal():
            raise ValueError(f"illegal card {card} for seat {p}")
        if self.trick:
            led = EFF[self.trump][self.trick[0][1]]
            if EFF[self.trump][card] != led:
                self.voids[p][led] = True
        self.hands[p].remove(card)
        self.trick.append((p, card))
        if len(self.trick) == self.n_players:
            w = self.winning()[0]
            self.tricks_won[w % 2] += 1
            self.history.append(self.trick)
            self.played_cards.extend(c for _, c in self.trick)
            self.trick = []
            self.trick_no += 1
            self.current = w
        else:
            self.current = self._next(p)

    def points(self):
        """[team0 points, team1 points] for the finished hand."""
        mt = self.maker % 2
        k = self.tricks_won[mt]
        pts = [0, 0]
        if k == 5:
            pts[mt] = 4 if self.alone else 2
        elif k >= 3:
            pts[mt] = 1
        else:
            pts[1 - mt] = 2
        return pts


def play_out(hands, contract, dealer, upcard, players, on_play=None):
    state = PlayState(hands, contract, dealer, upcard)
    while not state.done:
        p = state.current
        c = players[p].play(state)
        if on_play:
            on_play(state, p, c)
        state.play(c)
    return state


def play_full_hand(agents, rng=random, dealer=None, stick_the_dealer=False, deal_result=None):
    """Deal, bid and play one hand. Returns (contract, final PlayState) or (None, None) if all pass."""
    hands, upcard, _ = deal_result or deal(rng)
    if dealer is None:
        dealer = rng.randrange(4)
    contract, hands = run_bidding(hands, upcard, dealer, agents, stick_the_dealer)
    if contract is None:
        return None, None
    return contract, play_out(hands, contract, dealer, upcard, agents)
