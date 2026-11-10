"""The game behind the web UI, written as a step-by-step state machine.

There are no threads and no sleeping, so it runs the same way in a browser
(Pyodide) as it does in a test. The page drives it like this:

    game = Game()
    out = game.step()              # {"state": {...}, "wait_ms": 800 or None}
    # wait_ms is a number  -> show the state, wait that long, call step() again
    # wait_ms is None      -> the human has to decide: show state["pending"],
    #                         then call game.submit(answer) and step() again

You sit South (seat 0). North is your partner, West and East are the opponents.
"""
import os
import random
import time

from .cards import EFF, SUIT_NAMES, TRUMP_POWER, card_str, suit_of
from .engine import BidView, Bot, PlayState, deal, iter_bidding
from .heuristics import HeuristicAgent
from .mc_bidder import MonteCarloBidder
from .rl import RLPlayer

SEATS = ["You", "West", "Partner", "East"]


def load_player(model_path):
    """The trained card player if the model file is there, else the rule-based fallback."""
    if os.path.exists(model_path):
        try:
            return RLPlayer(model_path), True
        except Exception as e:  # e.g. a half-written file
            print("could not load model:", e)
    return HeuristicAgent(), False


def sort_hand(cards, trump=None):
    order = [0, 2, 1, 3]  # S H C D: alternate colours

    def key(c):
        if trump is not None:
            s = EFF[trump][c]
            return (0 if s == trump else 1 + order.index(s), -max(TRUMP_POWER[trump][c], c % 6))
        return (order.index(suit_of(c)), -(c % 6))
    return sorted(cards, key=key)


def res_json(results):
    return [{"label": r.option.label(), "ev": round(r.ev, 3), "se": round(r.se, 3),
             "p": None if r.p_make != r.p_make else round(r.p_make, 3)} for r in results]


def analyze_bid(req, player, has_model):
    """The bid analyzer tab: expected points for every option in a hand the user picked."""
    hand = [int(c) for c in req["hand"]]
    up = int(req["upcard"])
    if len(set(hand)) != 5 or up in hand:
        raise ValueError("pick exactly 5 hand cards and a different upcard")
    dealer = int(req.get("dealer", 0))
    rnd = int(req.get("round", 1))
    stick = bool(req.get("stick", False))
    sims = max(50, min(int(req.get("sims", 1000)), 5000))
    pos = (0 - dealer - 1) % 4
    view = BidView(0, hand, up, dealer, rnd, pos, must_call=stick and dealer == 0 and rnd == 2)
    t0 = time.time()
    res = MonteCarloBidder(sims, play_policy=player, stick_the_dealer=stick).analyze(view)
    return {"results": res_json(res), "sims": sims, "seconds": round(time.time() - t0, 2),
            "rollout": "RL model" if has_model else "heuristic"}


class Game:
    def __init__(self, player, has_model, sims=200, delay_ms=800, target=10, stick=False,
                 hint_sims=400, seed=None):
        self.delay, self.target, self.stick = delay_ms, target, stick
        self.rng = random.Random(seed)
        self.player = player
        self.hint_mc = MonteCarloBidder(hint_sims, play_policy=player, stick_the_dealer=stick)
        self.bots = {s: Bot(MonteCarloBidder(sims, play_policy=player, stick_the_dealer=stick, seed=s), player)
                     for s in (1, 2, 3)}
        self._answer = None
        self._answered = False
        self.v = {
            "phase": "starting", "score": [0, 0], "dealer": self.rng.randrange(4),
            "hand": [], "upcard": None, "upcard_state": "none", "bid_round": 1, "bids": {},
            "contract": None, "trick": [], "last_trick": None, "tricks_won": [0, 0],
            "counts": [5, 5, 5, 5], "turn": None, "thinking": None, "pending": None,
            "log": [], "result": None, "reveal": None, "has_model": has_model,
            "target": target, "hand_no": 0,
        }
        self._steps = self._run()
        self._finished = False

    # ---- driving the game from outside
    def step(self):
        """Advance to the next moment worth showing. Safe to call while waiting for the human."""
        wait = None
        if not self._waiting_for_human() and not self._finished:
            try:
                wait = next(self._steps)
            except StopIteration:
                self._finished = True
            except Exception:
                self._finished = True  # a crashed game can't resume; say so instead of looking finished
                raise
        return {"state": self.v, "wait_ms": wait}

    def submit(self, ans):
        """Give the human's answer to the pending question. Returns an error string, or None if accepted."""
        p = self.v["pending"]
        if not p or self._answered:
            return "no decision pending"
        if not isinstance(ans, dict):
            return "answer must be an object"
        if p["type"] == "play" and ans.get("card") not in p["legal"]:
            return "illegal card"
        if p["type"] == "discard" and ans.get("card") not in p["cards"]:
            return "not in hand"
        if p["type"] == "bid1" and ans.get("kind") not in ("order", "pass"):
            return "choose order or pass"
        if p["type"] == "bid2" and ans.get("suit") is not None and ans.get("suit") not in p["suits"]:
            return "not a legal suit"
        self._answer, self._answered = ans, True
        return None

    def _waiting_for_human(self):
        return self.v["pending"] is not None and not self._answered

    # ---- small helpers used by the game loop
    def _update(self, **kw):
        self.v.update(kw)

    def _log(self, msg):
        self.v["log"] = (self.v["log"] + [msg])[-60:]

    def _ask(self, pending):
        """Show a question, stop until submit() answers it, then return the answer."""
        self._update(pending=pending, turn=0, thinking=None)
        yield None
        self._update(pending=None)
        self._answered = False
        return self._answer

    def _think(self, seat):
        """Show 'thinking...' before the slow part, so the page can draw it first."""
        self._update(turn=seat, thinking=seat)
        yield 0

    # ---- the game loop
    def _run(self):
        while max(self.v["score"]) < self.target:
            yield from self._play_hand()
            yield from self._ask({"type": "next"})
            self._update(dealer=(self.v["dealer"] + 1) % 4)
        won = self.v["score"][0] >= self.target
        self._update(phase="game_over", result="You win the game!" if won else "The bots win the game.")
        yield from self._ask({"type": "new_game"})

    def _play_hand(self):
        hands, up, _ = deal(self.rng)
        d = self.v["dealer"]
        self._update(phase="bidding", hand=sort_hand(hands[0]), upcard=up, upcard_state="up", bid_round=1,
                     bids={}, contract=None, trick=[], last_trick=None, tricks_won=[0, 0],
                     counts=[5, 5, 5, 5], result=None, reveal=None, hand_no=self.v["hand_no"] + 1)
        self._log(f"--- Hand {self.v['hand_no']}: {SEATS[d]} deal{'' if d == 0 else 's'}, "
                  f"upcard {card_str(up)}")
        contract, h2 = yield from self._bidding(hands, up, d)
        if contract is None:
            self._log("Everyone passed - redeal.")
            self._update(phase="hand_over", result="Everyone passed. Redeal.", upcard_state="down",
                         reveal=[sort_hand(h) for h in hands])
            return
        t = contract.trump
        self._update(phase="play", contract={"trump": t, "trump_name": SUIT_NAMES[t], "maker": contract.maker,
                                             "alone": contract.alone},
                     upcard_state="taken" if contract.upcard_taken else "down",
                     hand=sort_hand(h2[0], t), bids={})
        self._log(f"{SEATS[contract.maker]} made {SUIT_NAMES[t]} trump{' ALONE' if contract.alone else ''}.")
        st = PlayState(h2, contract, d, up)
        start_hands = [sort_hand(h, t) for h in h2]
        counts = [len(h) for h in st.hands]
        if st.out >= 0:
            counts[st.out] = 0
        self._update(counts=counts)
        while not st.done:
            p = st.current
            c = yield from (self._human_play(st) if p == 0 else self._bot_play(p, st))
            n_before = len(st.trick)
            st.play(c)
            counts[p] -= 1
            trick = self.v["trick"] + [{"seat": p, "card": c}]
            self._update(trick=trick, counts=list(counts), turn=None, thinking=None,
                         hand=sort_hand(st.hands[0], t))
            if n_before == st.n_players - 1:
                w = st.current
                self._log(f"Trick {st.trick_no}: {SEATS[w]} win{'' if w == 0 else 's'} "
                          f"({' '.join(card_str(x['card']) for x in trick)})")
                self._update(trick_winner=w, tricks_won=list(st.tricks_won))
                yield int(self.delay * 1.6)
                self._update(trick=[], last_trick={"cards": trick, "winner": w}, trick_winner=None)
        pts = st.points()
        self.v["score"][0] += pts[0]
        self.v["score"][1] += pts[1]
        mt = contract.maker % 2
        k = st.tricks_won[mt]
        team = "You & Partner" if mt == 0 else "West & East"
        if k < 3:
            msg = f"{team} got euchred ({k} tricks)! +2 to {'You & Partner' if mt else 'West & East'}."
        else:
            msg = f"{team} took {k} tricks: +{max(pts)}."
        self._log(msg)
        self._update(phase="hand_over", result=msg, score=list(self.v["score"]), reveal=start_hands)

    # ---- bidding: engine.iter_bidding asks, we answer (for the human, after they decide)
    def _bidding(self, hands, up, dealer):
        gen = iter_bidding(hands, up, dealer, self.stick)
        try:
            req = next(gen)
            while True:
                answer = yield from self._decide(req)
                req = gen.send(answer)
        except StopIteration as done:
            return done.value

    def _decide(self, req):
        kind, view = req[0], req[1]
        human = view.player == 0
        if kind == "bid1":
            return (yield from (self._human_bid1 if human else self._bot_bid1)(view))
        if kind == "bid2":
            return (yield from (self._human_bid2 if human else self._bot_bid2)(view))
        return (yield from (self._human_discard if human else self._bot_discard)(view, *req[2:]))

    # -- bots
    def _bot_said(self, seat, text):
        bids = dict(self.v["bids"])
        bids[str(seat)] = text
        self._update(bids=bids, thinking=None)
        self._log(f"{SEATS[seat]}: {text}")
        yield self.delay

    def _bot_bid1(self, view):
        self._update(bid_round=1)
        yield from self._think(view.player)
        go, alone = self.bots[view.player].bid_round1(view)
        yield from self._bot_said(view.player, ("Order up" + (" ALONE" if alone else "")) if go else "Pass")
        return go, alone

    def _bot_bid2(self, view):
        if self.v["bid_round"] == 1:
            self._update(bid_round=2, upcard_state="down", bids={})
        yield from self._think(view.player)
        s, alone = self.bots[view.player].bid_round2(view)
        yield from self._bot_said(view.player, (SUIT_NAMES[s] + (" ALONE" if alone else "")) if s is not None else "Pass")
        return s, alone

    def _bot_discard(self, view, hand6, trump, maker, alone):
        yield from self._think(view.player)
        c = self.bots[view.player].discard(view, hand6, trump, maker, alone)
        self._log(f"{SEATS[view.player]} picks up {card_str(view.upcard)} and discards.")
        self._update(thinking=None)
        return c

    def _bot_play(self, seat, state):
        yield from self._think(seat)
        c = self.bots[seat].play(state)
        yield self.delay
        return c

    # -- the human
    def _human_said(self, text):
        bids = dict(self.v["bids"])
        bids["0"] = text
        self._update(bids=bids)
        self._log(f"You: {text}")

    def _human_bid1(self, view):
        self._update(bid_round=1, thinking=0, turn=0)
        yield 0  # let the page show "Simulating..." before the slow analysis
        analysis = res_json(self.hint_mc.analyze(view))
        a = yield from self._ask({"type": "bid1", "suit": suit_of(view.upcard), "dealer_is_you": view.dealer == 0,
                                  "partner_deals": view.dealer == 2, "analysis": analysis})
        go = a.get("kind") == "order"
        alone = go and bool(a.get("alone"))
        self._human_said(("Order up" + (" ALONE" if alone else "")) if go else "Pass")
        return go, alone

    def _human_bid2(self, view):
        if self.v["bid_round"] == 1:
            self._update(bid_round=2, upcard_state="down", bids={})
        self._update(thinking=0, turn=0)
        yield 0
        analysis = res_json(self.hint_mc.analyze(view))
        up = suit_of(view.upcard)
        a = yield from self._ask({"type": "bid2", "suits": [s for s in range(4) if s != up],
                                  "must_call": view.must_call, "analysis": analysis})
        s = a.get("suit")
        if s is None or s == up or s not in range(4):
            if view.must_call:
                s = [x for x in range(4) if x != up][0]
            else:
                self._human_said("Pass")
                return None, False
        alone = bool(a.get("alone"))
        self._human_said(SUIT_NAMES[s] + (" ALONE" if alone else ""))
        return s, alone

    def _human_discard(self, view, hand6, trump, maker, alone):
        self._update(hand=sort_hand(hand6, trump), thinking=0, turn=0)
        yield 0
        res = self.hint_mc.analyze_discard(view, hand6, trump, maker, alone, 200)
        a = yield from self._ask({"type": "discard", "cards": list(hand6), "analysis": res_json(res),
                                  "best": res[0].option.discard})
        self._log(f"You pick up {card_str(view.upcard)} and discard {card_str(a['card'])}.")
        return a["card"]

    def _human_play(self, state):
        legal = state.legal()
        if isinstance(self.player, RLPlayer):
            q = {str(c): round(v, 3) for c, v in self.player.q_values(state).items()}
        else:
            best = self.player.play(state)
            q = {str(c): (1.0 if c == best else 0.0) for c in legal}
        a = yield from self._ask({"type": "play", "legal": legal, "q": q,
                                  "q_is_model": isinstance(self.player, RLPlayer)})
        return a["card"]
