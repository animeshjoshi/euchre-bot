"""Train the RL card player by self-play (Monte Carlo control with a Q-network).

Each hand: deal, bid with the heuristic bidder (so contracts look like real ones), then play.
Seats controlled by the learner choose epsilon-greedy w.r.t. Q; every decision (s, a) is
regressed onto the final signed hand points G for that seat's team:
        loss = (Q(s, a) - G)^2
Half the hands are pure self-play, half are learner vs heuristic opponents.

    python train.py --hands 200000 --out models/euchre_q.npz
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

from euchre.cards import CANON
from euchre.engine import Bot, deal, play_out, run_bidding
from euchre.evaluate import duplicate_match
from euchre.heuristics import HeuristicAgent, RandomAgent
from euchre.rl import FEATURE_DIM, N_ACTIONS, NumpyMLP, RLPlayer, encode


def make_net(hidden):
    layers, d = [], FEATURE_DIM
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, N_ACTIONS))
    return nn.Sequential(*layers)


def to_numpy(net):
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return NumpyMLP([(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in lin])


class Learner:
    """Records (features, action) for learner seats while playing."""

    def __init__(self, player):
        self.player = player
        self.records = []

    def play(self, state):
        legal = state.legal()
        if len(legal) == 1:
            return legal[0]
        x = encode(state)
        c = self.player.play(state)
        self.records.append((state.current, x, CANON[state.trump][c]))
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=200_000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--train-every", type=int, default=8, help="hands between gradient steps")
    ap.add_argument("--eps-start", type=float, default=0.3)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-deals", type=int, default=2000)
    ap.add_argument("--out", default="models/euchre_q.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)

    net = make_net(args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    X = np.zeros((args.buffer, FEATURE_DIM), np.float32)
    A = np.zeros(args.buffer, np.int64)
    G = np.zeros(args.buffer, np.float32)
    size = ptr = 0

    heur = HeuristicAgent()
    rl = RLPlayer(to_numpy(net), epsilon=args.eps_start, rng=nrng)
    learner = Learner(rl)
    bidders = [heur] * 4
    best_score, t0, losses = -1e9, time.time(), []

    for hand_i in range(1, args.hands + 1):
        frac = min(hand_i / (0.7 * args.hands), 1.0)
        rl.epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        hands, upcard, _ = deal(rng)
        dealer = rng.randrange(4)
        contract, hands = run_bidding(hands, upcard, dealer, bidders)
        if contract is None:
            continue
        mode = rng.random()
        if mode < 0.5:
            players = [learner] * 4
        else:
            t = rng.randrange(2)
            players = [learner if p % 2 == t else heur for p in range(4)]
        learner.records.clear()
        pts = play_out(hands, contract, dealer, upcard, players).points()
        for p, x, a in learner.records:
            X[ptr], A[ptr], G[ptr] = x, a, pts[p % 2] - pts[1 - p % 2]
            ptr = (ptr + 1) % args.buffer
            size = min(size + 1, args.buffer)

        if hand_i % args.train_every == 0 and size >= 5000:
            idx = nrng.integers(0, size, args.batch)
            xb = torch.from_numpy(X[idx])
            ab = torch.from_numpy(A[idx])
            gb = torch.from_numpy(G[idx])
            q = net(xb).gather(1, ab[:, None]).squeeze(1)
            loss = nn.functional.mse_loss(q, gb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            rl.net = to_numpy(net)

        if hand_i % args.eval_every == 0 or hand_i == args.hands:
            greedy = RLPlayer(to_numpy(net))
            m, se = duplicate_match(Bot(heur, greedy), Bot(heur, heur), args.eval_deals, seed=12345)
            lm = np.mean(losses[-2000:]) if losses else float("nan")
            print(f"hand {hand_i:>8}  eps {rl.epsilon:.3f}  loss {lm:.3f}  "
                  f"vs heuristic {m:+.3f} +/- {se:.3f} pts/hand  ({time.time() - t0:.0f}s)", flush=True)
            if m > best_score:
                best_score = m
                greedy.net.save(args.out)
    print(f"saved best model ({best_score:+.3f} pts/hand vs heuristic) -> {args.out}")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()
