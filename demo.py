"""Euchre bot command line.

Analyse a bidding decision (you are seat 0 / South):
    python demo.py bid --hand "JH JD AH 9C KS" --upcard QH --dealer right
    python demo.py bid --hand "JH JD AH 9C KS" --upcard QS --dealer me --round 2 --sims 2000

Watch the bot (MC bidding + RL card play) play a full hand, with its reasoning:
    python demo.py hand --seed 7
"""
import argparse
import os
import random

from euchre.cards import SUITS, card_str, cards_str, parse_cards, parse_card
from euchre.engine import BidView, Bot, deal, play_out, run_bidding
from euchre.heuristics import HeuristicAgent
from euchre.mc_bidder import MonteCarloBidder
from euchre.rl import RLPlayer

SEAT = ["South (you)", "West", "North (partner)", "East"]
DEALER = {"me": 0, "left": 1, "partner": 2, "right": 3}
DEFAULT_MODEL = "models/euchre_q.npz"


def load_player(path):
    if path and os.path.exists(path):
        return RLPlayer(path)
    print(f"[no RL model at {path!r}; using heuristic card play in rollouts]")
    return HeuristicAgent()


def cmd_bid(a):
    hand = parse_cards(a.hand)
    up = parse_card(a.upcard)
    if len(hand) != 5 or up in hand:
        raise SystemExit("need 5 distinct cards and an upcard not in the hand")
    dealer = DEALER[a.dealer]
    pos = (0 - dealer - 1) % 4
    v = BidView(0, hand, up, dealer, a.round, pos, must_call=a.stick and dealer == 0 and a.round == 2)
    mc = MonteCarloBidder(n_sims=a.sims, play_policy=load_player(a.model),
                          stick_the_dealer=a.stick, seed=a.seed)
    print(f"Hand {cards_str(hand)} | upcard {card_str(up)}"
          f"{' (turned down)' if a.round == 2 else ''} | dealer: {SEAT[dealer]} | round {a.round}, "
          f"you bid {pos + 1}{'st' if pos == 0 else 'nd' if pos == 1 else 'rd' if pos == 2 else 'th'}\n")
    res = mc.analyze(v)
    for r in res:
        print("  " + str(r))
    print(f"\n=> best: {res[0].option.label()}   (EV in signed points for your team, {a.sims} simulated deals)")


def cmd_hand(a):
    rng = random.Random(a.seed)
    player = load_player(a.model)
    mc = MonteCarloBidder(n_sims=a.sims, play_policy=player, seed=a.seed)
    bot = Bot(mc, player)
    hands, up, _ = deal(rng)
    dealer = rng.randrange(4)
    print(f"Dealer: {SEAT[dealer]}   upcard: {card_str(up)}")
    for p in range(4):
        print(f"  {SEAT[p]:<16} {cards_str(sorted(hands[p]))}")

    class Narrator(Bot):
        def bid_round1(self, v):
            r = super().bid_round1(v)
            self._show(v)
            return r

        def bid_round2(self, v):
            r = super().bid_round2(v)
            self._show(v)
            return r

        def _show(self, v):
            print(f"\n{SEAT[v.player]} (round {v.round}) analysis:")
            for res in mc.last_analysis[1][:4]:
                print("    " + str(res))
            print(f"  -> {mc.last_analysis[1][0].option.label()}")

    narr = Narrator(mc, player)
    contract, h2 = run_bidding(hands, up, dealer, [narr] * 4)
    if contract is None:
        print("\nAll passed - redeal.")
        return
    print(f"\nTrump {SUITS[contract.trump]}, maker {SEAT[contract.maker]}"
          f"{' ALONE' if contract.alone else ''}"
          f"{', dealer discarded ' + card_str(contract.discard) if contract.discard is not None else ''}\n")

    def on_play(state, p, c):
        if not state.trick:
            print(f"-- trick {state.trick_no + 1}")
        qs = player.q_values(state) if isinstance(player, RLPlayer) else {}
        qtxt = "  ".join(f"{card_str(k)}:{v:+.2f}" for k, v in sorted(qs.items(), key=lambda kv: -kv[1]))
        print(f"   {SEAT[p]:<16} plays {card_str(c)}    {'Q: ' + qtxt if len(qs) > 1 else ''}")

    st = play_out(h2, contract, dealer, up, [bot] * 4, on_play)
    pts = st.points()
    print(f"\nTricks: N/S {st.tricks_won[0]}  E/W {st.tricks_won[1]}   points: N/S {pts[0]}  E/W {pts[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bid")
    b.add_argument("--hand", required=True)
    b.add_argument("--upcard", required=True)
    b.add_argument("--dealer", choices=DEALER, required=True)
    b.add_argument("--round", type=int, choices=(1, 2), default=1)
    b.add_argument("--stick", action="store_true", help="stick-the-dealer rule")
    b.add_argument("--sims", type=int, default=1000)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("hand")
    h.add_argument("--seed", type=int, default=None)
    h.add_argument("--sims", type=int, default=300)
    h.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    cmd_bid(a) if a.cmd == "bid" else cmd_hand(a)


if __name__ == "__main__":
    main()
