"""Local web UI for the Euchre bot (stdlib only).

    python server.py            -> open http://localhost:8000

You sit South; West/East are bot opponents, North is your bot partner.
Bots bid with Monte Carlo and play with the RL Q-network (models/euchre_q.npz),
falling back to heuristic play if no model has been trained yet.
"""
import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from euchre.cards import EFF, RANKS, SUITS, SUIT_NAMES, TRUMP_POWER, card_str, suit_of
from euchre.engine import Agent, BidView, Bot, PlayState, deal, run_bidding
from euchre.heuristics import HeuristicAgent
from euchre.mc_bidder import MonteCarloBidder
from euchre.rl import RLPlayer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "euchre_q.npz")
SEATS = ["You", "West", "Partner", "East"]


class Cancelled(Exception):
    pass


def load_player():
    if os.path.exists(MODEL):
        try:
            return RLPlayer(MODEL), True
        except Exception as e:  # file mid-write during training, etc.
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


class Game:
    def __init__(self, sims=200, delay=0.8, target=10, stick=False, hint_sims=400):
        self.lock = threading.Lock()
        self.cancel = False
        self.delay, self.target, self.stick = delay, target, stick
        self.rng = random.Random()
        self.player, self.has_model = load_player()
        self.hint_mc = MonteCarloBidder(hint_sims, play_policy=self.player, stick_the_dealer=stick)
        bots = [Bot(MonteCarloBidder(sims, play_policy=self.player, stick_the_dealer=stick, seed=i),
                    self.player) for i in range(4)]
        self.agents = [Human(self)] + [UIBot(self, s, bots[s]) for s in (1, 2, 3)]
        self.answer = None
        self.answer_event = threading.Event()
        self.v = {
            "version": 0, "phase": "starting", "score": [0, 0], "dealer": self.rng.randrange(4),
            "hand": [], "upcard": None, "upcard_state": "none", "bid_round": 1, "bids": {},
            "contract": None, "trick": [], "last_trick": None, "tricks_won": [0, 0],
            "counts": [5, 5, 5, 5], "turn": None, "thinking": None, "pending": None,
            "log": [], "result": None, "reveal": None, "has_model": self.has_model,
            "target": target, "hand_no": 0,
        }
        threading.Thread(target=self._run, daemon=True).start()

    # ---- state helpers
    def update(self, **kw):
        with self.lock:
            self.v.update(kw)
            self.v["version"] += 1

    def log(self, msg):
        with self.lock:
            self.v["log"] = (self.v["log"] + [msg])[-60:]
            self.v["version"] += 1

    def snapshot(self):
        with self.lock:
            return json.dumps(self.v)

    def sleep(self, t):
        end = time.time() + t
        while time.time() < end:
            if self.cancel:
                raise Cancelled
            time.sleep(0.05)

    def ask(self, pending):
        self.answer_event.clear()
        self.update(pending=pending, turn=0, thinking=None)
        while not self.answer_event.wait(0.1):
            if self.cancel:
                raise Cancelled
        self.update(pending=None)
        return self.answer

    def submit(self, ans):
        p = self.v["pending"]
        if not p:
            return "no decision pending"
        if p["type"] == "play" and ans.get("card") not in p["legal"]:
            return "illegal card"
        if p["type"] == "discard" and ans.get("card") not in p["cards"]:
            return "not in hand"
        self.answer = ans
        self.answer_event.set()
        return None

    # ---- game loop
    def _run(self):
        try:
            while max(self.v["score"]) < self.target:
                self._play_hand()
                self.ask({"type": "next"})
                self.update(dealer=(self.v["dealer"] + 1) % 4)
            won = self.v["score"][0] >= self.target
            self.update(phase="game_over", result="You win the game!" if won else "The bots win the game.")
            self.ask({"type": "new_game"})
        except Cancelled:
            pass

    def _play_hand(self):
        hands, up, _ = deal(self.rng)
        d = self.v["dealer"]
        self.update(phase="bidding", hand=sort_hand(hands[0]), upcard=up, upcard_state="up", bid_round=1,
                    bids={}, contract=None, trick=[], last_trick=None, tricks_won=[0, 0],
                    counts=[5, 5, 5, 5], result=None, reveal=None, hand_no=self.v["hand_no"] + 1)
        self.log(f"--- Hand {self.v['hand_no']}: {SEATS[d]} deal{'' if d == 0 else 's'}, "
                 f"upcard {card_str(up)}")
        contract, h2 = run_bidding(hands, up, d, self.agents, self.stick)
        if contract is None:
            self.log("Everyone passed - redeal.")
            self.update(phase="hand_over", result="Everyone passed. Redeal.", upcard_state="down",
                        reveal=[sort_hand(h) for h in hands])
            return
        t = contract.trump
        self.update(phase="play", contract={"trump": t, "trump_name": SUIT_NAMES[t], "maker": contract.maker,
                                            "alone": contract.alone},
                    upcard_state="taken" if contract.upcard_taken else "down",
                    hand=sort_hand(h2[0], t), bids={})
        self.log(f"{SEATS[contract.maker]} made {SUIT_NAMES[t]} trump{' ALONE' if contract.alone else ''}.")
        st = PlayState(h2, contract, d, up)
        start_hands = [sort_hand(h, t) for h in h2]
        counts = [len(h) for h in st.hands]
        if st.out >= 0:
            counts[st.out] = 0
        self.update(counts=counts)
        while not st.done:
            p = st.current
            c = self.agents[p].play(st)
            n_before = len(st.trick)
            st.play(c)
            counts[p] -= 1
            trick = self.v["trick"] + [{"seat": p, "card": c}]
            self.update(trick=trick, counts=list(counts), turn=None, thinking=None,
                        hand=sort_hand(st.hands[0], t))
            if n_before == st.n_players - 1:
                w = st.current
                self.log(f"Trick {st.trick_no}: {SEATS[w]} win{'' if w == 0 else 's'} "
                         f"({' '.join(card_str(x['card']) for x in trick)})")
                self.update(trick_winner=w, tricks_won=list(st.tricks_won))
                self.sleep(self.delay * 1.6)
                self.update(trick=[], last_trick={"cards": trick, "winner": w}, trick_winner=None)
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
        self.log(msg)
        self.update(phase="hand_over", result=msg, score=list(self.v["score"]), reveal=start_hands)


class UIBot(Agent):
    def __init__(self, game, seat, bot):
        self.g, self.seat, self.bot = game, seat, bot

    def _think(self):
        self.g.update(turn=self.seat, thinking=self.seat)

    def _bid(self, text):
        bids = dict(self.g.v["bids"])
        bids[str(self.seat)] = text
        self.g.update(bids=bids, thinking=None)
        self.g.log(f"{SEATS[self.seat]}: {text}")
        self.g.sleep(self.g.delay)

    def bid_round1(self, v):
        self.g.update(bid_round=1)
        self._think()
        go, alone = self.bot.bid_round1(v)
        self._bid(("Order up" + (" ALONE" if alone else "")) if go else "Pass")
        return go, alone

    def bid_round2(self, v):
        if self.g.v["bid_round"] == 1:
            self.g.update(bid_round=2, upcard_state="down", bids={})
        self._think()
        s, alone = self.bot.bid_round2(v)
        self._bid((SUIT_NAMES[s] + (" ALONE" if alone else "")) if s is not None else "Pass")
        return s, alone

    def discard(self, v, hand6, trump, maker, alone):
        self._think()
        c = self.bot.discard(v, hand6, trump, maker, alone)
        self.g.log(f"{SEATS[self.seat]} picks up {card_str(v.upcard)} and discards.")
        return c

    def play(self, state):
        self._think()
        c = self.bot.play(state)
        self.g.sleep(self.g.delay)
        return c


class Human(Agent):
    def __init__(self, game):
        self.g = game

    def _bid_text(self, text):
        bids = dict(self.g.v["bids"])
        bids["0"] = text
        self.g.update(bids=bids)
        self.g.log(f"You: {text}")

    def bid_round1(self, v):
        self.g.update(bid_round=1, thinking=0, turn=0)
        analysis = res_json(self.g.hint_mc.analyze(v))
        a = self.g.ask({"type": "bid1", "suit": suit_of(v.upcard), "dealer_is_you": v.dealer == 0,
                        "partner_deals": v.dealer == 2, "analysis": analysis})
        go = a.get("kind") == "order"
        alone = go and bool(a.get("alone"))
        self._bid_text(("Order up" + (" ALONE" if alone else "")) if go else "Pass")
        return go, alone

    def bid_round2(self, v):
        if self.g.v["bid_round"] == 1:
            self.g.update(bid_round=2, upcard_state="down", bids={})
        self.g.update(thinking=0, turn=0)
        analysis = res_json(self.g.hint_mc.analyze(v))
        up = suit_of(v.upcard)
        a = self.g.ask({"type": "bid2", "suits": [s for s in range(4) if s != up],
                        "must_call": v.must_call, "analysis": analysis})
        s = a.get("suit")
        if s is None or s == up or s not in range(4):
            if v.must_call:
                s = [x for x in range(4) if x != up][0]
            else:
                self._bid_text("Pass")
                return None, False
        alone = bool(a.get("alone"))
        self._bid_text(SUIT_NAMES[s] + (" ALONE" if alone else ""))
        return s, alone

    def discard(self, v, hand6, trump, maker, alone):
        self.g.update(hand=sort_hand(hand6, trump), thinking=0, turn=0)
        res = self.g.hint_mc.analyze_discard(v, hand6, trump, maker, alone, 200)
        a = self.g.ask({"type": "discard", "cards": list(hand6), "analysis": res_json(res),
                        "best": res[0].option.discard})
        self.g.log(f"You pick up {card_str(v.upcard)} and discard {card_str(a['card'])}.")
        return a["card"]

    def play(self, state):
        legal = state.legal()
        q = {}
        if isinstance(self.g.player, RLPlayer):
            q = {str(c): round(v, 3) for c, v in self.g.player.q_values(state).items()}
        else:
            best = self.g.player.play(state)
            q = {str(c): (1.0 if c == best else 0.0) for c in legal}
        a = self.g.ask({"type": "play", "legal": legal, "q": q, "q_is_model": isinstance(self.g.player, RLPlayer)})
        return a["card"]


# ------------------------------------------------------------------ HTTP --
GAME = {"g": None}
GAME_LOCK = threading.Lock()


def new_game(opts):
    with GAME_LOCK:
        if GAME["g"]:
            GAME["g"].cancel = True
        GAME["g"] = Game(sims=int(opts.get("sims", 200)), delay=float(opts.get("delay", 0.8)),
                         target=int(opts.get("target", 10)), stick=bool(opts.get("stick", False)))
    return GAME["g"]


def analyze(req):
    hand = [int(c) for c in req["hand"]]
    up = int(req["upcard"])
    if len(set(hand)) != 5 or up in hand:
        raise ValueError("pick exactly 5 hand cards and a different upcard")
    dealer = int(req.get("dealer", 0))
    rnd = int(req.get("round", 1))
    stick = bool(req.get("stick", False))
    sims = max(50, min(int(req.get("sims", 1000)), 5000))
    player, has_model = load_player()
    pos = (0 - dealer - 1) % 4
    v = BidView(0, hand, up, dealer, rnd, pos, must_call=stick and dealer == 0 and rnd == 2)
    t0 = time.time()
    res = MonteCarloBidder(sims, play_policy=player, stick_the_dealer=stick).analyze(v)
    return {"results": res_json(res), "sims": sims, "seconds": round(time.time() - t0, 2),
            "rollout": "RL model" if has_model else "heuristic"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "web", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/api/state":
            g = GAME["g"]
            return self._send(200, g.snapshot() if g else '{"phase":"none"}')
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/new":
                return self._send(200, new_game(body).snapshot())
            if self.path == "/api/act":
                g = GAME["g"]
                err = g.submit(body) if g else "no game"
                return self._send(400 if err else 200, json.dumps({"error": err} if err else {"ok": True}))
            if self.path == "/api/analyze":
                return self._send(200, json.dumps(analyze(body)))
            self._send(404, '{"error":"not found"}')
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Euchre UI at http://localhost:{a.port}  (model: {'RL' if os.path.exists(MODEL) else 'heuristic'})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
