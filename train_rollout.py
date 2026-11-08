"""Train the RL card player by rollout-based approximate policy iteration.

Plain Monte Carlo Q-learning (train.py) regresses each move onto one noisy final score.
Here every training state gets a low-variance target for *every* legal card:

  for K determinizations (random deals of the hidden cards, consistent with the cards seen,
  known voids, the dealer holding the picked-up upcard, and hand sizes):
      for each legal card a: play a, then finish the hand with the current rollout policy
  target Q(s, a) = mean signed points for my team

The network is trained on those targets, becomes the rollout policy of the next iteration
(policy improvement), and so on. Iteration 0 rolls out with the heuristic player.
Rollouts run in parallel worker processes.

    python train_rollout.py --iters 4 --states 60000 --k 16
"""
import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from euchre.cards import ALL_CARDS, CANON, EFF
from euchre.engine import Bot, PlayState, deal, run_bidding
from euchre.evaluate import duplicate_match
from euchre.heuristics import HeuristicAgent
from euchre.rl import FEATURE_DIM, N_ACTIONS, NumpyMLP, RLPlayer, encode


# ------------------------------------------------------------ determinize --
def determinize(state, me, rng, known_extra=(), tries=30):
    """Sample hidden hands for the other seats consistent with what `me` has seen."""
    t = state.trump
    known = set(state.hands[me]) | set(state.played_cards) | {c for _, c in state.trick} | set(known_extra)
    forced = {}
    # the dealer picked up the upcard: if not yet played it is in the dealer's hand
    if (state.upcard_taken and state.dealer != me and state.upcard not in known
            and not state.voids[state.dealer][t]):
        forced[state.dealer] = state.upcard
    known.add(state.upcard)
    pool = [c for c in ALL_CARDS if c not in known]
    others = [p for p in range(4) if p != me and p != state.out]
    need = {p: len(state.hands[p]) - (1 if forced.get(p) is not None else 0) for p in others}
    for attempt in range(tries):
        rng.shuffle(pool)
        avail = list(pool)
        hands = [None] * 4
        hands[me] = list(state.hands[me])
        if state.out >= 0:
            hands[state.out] = []
        ok = True
        use_voids = attempt < tries - 1
        # most-constrained seat first
        for p in sorted(others, key=lambda q: -sum(state.voids[q])):
            v = state.voids[p]
            elig = [c for c in avail if not (use_voids and v[EFF[t][c]])]
            if len(elig) < need[p]:
                ok = False
                break
            h = elig[:need[p]]
            for c in h:
                avail.remove(c)
            if p in forced:
                h = h + [forced[p]]
            hands[p] = h
        if ok:
            return hands
    return None


def finish(state, policy):
    while not state.done:
        state.play(policy.play(state))
    pts = state.points()
    return pts


# ----------------------------------------------------------------- worker --
_W = {}


def _init(layers, seed):
    _W["rng"] = random.Random(seed + os.getpid())
    _W["policy"] = RLPlayer(NumpyMLP(layers)) if layers else HeuristicAgent()
    _W["heur"] = HeuristicAgent()


def _gen(args):
    """Play hands with the rollout policy; label sampled decision points. Returns arrays."""
    n_states, k, eps = args
    rng, pol = _W["rng"], _W["policy"]
    heur = _W["heur"]
    X, M, Y = [], [], []
    while len(X) < n_states:
        hands, up, _ = deal(rng)
        dealer = rng.randrange(4)
        contract, hands = run_bidding(hands, up, dealer, [heur] * 4)
        if contract is None:
            continue
        st = PlayState(hands, contract, dealer, up)
        while not st.done and len(X) < n_states:
            me = st.current
            legal = st.legal()
            if len(legal) > 1 and rng.random() < 0.5:
                extra = (contract.discard,) if contract.discard is not None and me == dealer else ()
                sums = np.zeros(N_ACTIONS, np.float32)
                cnt = 0
                for _ in range(k):
                    hs = determinize(st, me, rng, extra)
                    if hs is None:
                        continue
                    cnt += 1
                    for a in legal:
                        s2 = st.clone(hs)
                        s2.play(a)
                        pts = finish(s2, pol)
                        sums[CANON[st.trump][a]] += pts[me % 2] - pts[1 - me % 2]
                if cnt:
                    mask = np.zeros(N_ACTIONS, bool)
                    for a in legal:
                        mask[CANON[st.trump][a]] = True
                    X.append(encode(st, me))
                    M.append(mask)
                    Y.append(sums / cnt)
            # behaviour: rollout policy with some exploration so states are diverse
            c = rng.choice(legal) if rng.random() < eps else pol.play(st)
            st.play(c)
    return np.array(X), np.array(M), np.array(Y)


# ------------------------------------------------------------------ train --
def make_net(hidden):
    layers, d = [], FEATURE_DIM
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, N_ACTIONS))
    return nn.Sequential(*layers)


def to_layers(net):
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy())
            for m in net if isinstance(m, nn.Linear)]


def fit(net, X, M, Y, epochs, lr, batch=512, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Mt, Yt = map(torch.from_numpy, (X, M.astype(np.float32), Y))
    n = len(X)
    perm = torch.randperm(n)
    nval = n // 20
    vi, ti = perm[:nval], perm[nval:]
    for ep in range(epochs):
        idx = ti[torch.randperm(len(ti))]
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            err = (net(Xt[b]) - Yt[b]) * Mt[b]
            loss = (err ** 2).sum() / Mt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            err = (net(Xt[vi]) - Yt[vi]) * Mt[vi]
            vl = ((err ** 2).sum() / Mt[vi].sum()).item()
            # how often the net picks the target-best legal card
            q = net(Xt[vi]).masked_fill(Mt[vi] == 0, -1e9)
            yb = Yt[vi].masked_fill(Mt[vi] == 0, -1e9)
            agree = (q.argmax(1) == yb.argmax(1)).float().mean().item()
        print(f"    epoch {ep + 1:>2}  val mse {vl:.4f}  picks target-best {agree:.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--states", type=int, default=60_000, help="labelled states per iteration")
    ap.add_argument("--k", type=int, default=16, help="determinizations per state")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--keep", type=int, default=2, help="iterations of data to keep training on")
    ap.add_argument("--eval-deals", type=int, default=3000)
    ap.add_argument("--out", default="models/euchre_q.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.set_num_threads(4)

    heur = HeuristicAgent()
    layers = None            # iteration 0: heuristic rollouts
    net = make_net(a.hidden)
    data = []
    best = -1e9
    for it in range(a.iters):
        t0 = time.time()
        per = 500
        jobs = [(per, a.k, a.eps)] * (a.states // per)
        with mp.Pool(a.workers, initializer=_init, initargs=(layers, a.seed * 1000 + it)) as pool:
            parts = pool.map(_gen, jobs, chunksize=1)
        X = np.concatenate([p[0] for p in parts])
        M = np.concatenate([p[1] for p in parts])
        Y = np.concatenate([p[2] for p in parts])
        data = (data + [(X, M, Y)])[-a.keep:]
        Xa, Ma, Ya = (np.concatenate([d[i] for d in data]) for i in range(3))
        print(f"iter {it}: labelled {len(X)} states with "
              f"{'heuristic' if layers is None else 'network'} rollouts in {time.time() - t0:.0f}s; "
              f"training on {len(Xa)}", flush=True)
        fit(net, Xa, Ma, Ya, a.epochs, a.lr, seed=it)
        layers = to_layers(net)
        player = RLPlayer(NumpyMLP(layers))
        m, se = duplicate_match(Bot(heur, player), Bot(heur, heur), a.eval_deals, seed=4242)
        print(f"iter {it}: greedy net vs heuristic {m:+.3f} +/- {se:.3f} pts/hand "
              f"({time.time() - t0:.0f}s total)", flush=True)
        if m > best:
            best = m
            player.net.save(a.out)
            print(f"  saved -> {a.out}", flush=True)
    print(f"best {best:+.3f} pts/hand vs heuristic")


if __name__ == "__main__":
    main()
