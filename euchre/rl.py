"""Reinforcement-learning card player (inference side; training lives in train.py).

The player is a Q-network  Q(s, a)  ->  expected signed hand points for my team,
where a is one of the 24 cards. Suits are relabelled so trump is always suit 0 and the
same-colour suit is suit 1, so one network covers all four trump suits.
Inference uses only numpy; torch is needed only for training.
"""
import numpy as np

from .cards import CANON, EFF, NUM_CARDS, TRUMP_POWER, power

# SUIT_CARDS[t][s]: cards whose effective suit is s when t is trump
SUIT_CARDS = [[[c for c in range(NUM_CARDS) if EFF[t][c] == s] for s in range(4)] for t in range(4)]
from .engine import Agent
from .heuristics import HeuristicAgent

# feature layout (all relative to the acting seat `me`; seats r = (p - me) % 4)
_F = {}
_off = 0
for _name, _size in [("hand", 24), ("trick", 72), ("played", 24), ("led", 4), ("voids", 12),
                     ("maker", 4), ("alone", 1), ("tricks", 2), ("trick_no", 5), ("pos", 4),
                     ("winner", 4), ("upcard", 24), ("taken", 1), ("dealer", 4),
                     # derived card-tracking features (per canonical card in my hand)
                     ("boss", 24), ("higher", 24), ("beats", 24), ("unseen_suit", 4)]:
    _F[_name] = _off
    _off += _size
FEATURE_DIM = _off
N_ACTIONS = 24


def encode(state, me=None):
    """Feature vector of the public + own-hand information available to seat `me`."""
    if me is None:
        me = state.current
    t = state.trump
    cm = CANON[t]
    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    o = _F["hand"]
    for c in state.hands[me]:
        x[o + cm[c]] = 1.0
    o = _F["trick"]
    for p, c in state.trick:
        x[o + ((p - me) % 4 - 1) * 24 + cm[c]] = 1.0
    o = _F["played"]
    for c in state.played_cards:
        x[o + cm[c]] = 1.0
    if state.trick:
        x[_F["led"] + (EFF[t][state.trick[0][1]] ^ t)] = 1.0
        x[_F["winner"] + (state.winning()[0] - me) % 4] = 1.0
    o = _F["voids"]
    for r in (1, 2, 3):
        vp = state.voids[(me + r) % 4]
        for s in range(4):
            if vp[s]:
                x[o + (r - 1) * 4 + (s ^ t)] = 1.0
    x[_F["maker"] + (state.maker - me) % 4] = 1.0
    x[_F["alone"]] = float(state.alone)
    x[_F["tricks"]] = state.tricks_won[me % 2] / 5.0
    x[_F["tricks"] + 1] = state.tricks_won[1 - me % 2] / 5.0
    x[_F["trick_no"] + min(state.trick_no, 4)] = 1.0
    x[_F["pos"] + len(state.trick)] = 1.0
    x[_F["upcard"] + cm[state.upcard]] = 1.0
    x[_F["taken"]] = float(state.upcard_taken)
    x[_F["dealer"] + (state.dealer - me) % 4] = 1.0

    # card tracking: which cards could still be in someone else's hand
    seen = [False] * NUM_CARDS
    for c in state.hands[me]:
        seen[c] = True
    for c in state.played_cards:
        seen[c] = True
    for _, c in state.trick:
        seen[c] = True
    if not state.upcard_taken:
        seen[state.upcard] = True           # turned down: out of play
    sc = SUIT_CARDS[t]
    for s in range(4):
        x[_F["unseen_suit"] + (s ^ t)] = sum(not seen[c] for c in sc[s]) / 7.0
    led = EFF[t][state.trick[0][1]] if state.trick else None
    wpow = power(state.winning()[1], t, led) if state.trick else -1
    for c in state.hands[me]:
        s = EFF[t][c]
        pc = power(c, t, s)
        n_hi = sum(1 for u in sc[s] if not seen[u] and power(u, t, s) > pc)
        k = cm[c]
        x[_F["boss"] + k] = float(n_hi == 0)
        x[_F["higher"] + k] = n_hi / 6.0
        if state.trick and power(c, t, led) > wpow:
            x[_F["beats"] + k] = 1.0
    return x


def legal_mask(state):
    cm = CANON[state.trump]
    m = np.zeros(N_ACTIONS, dtype=bool)
    for c in state.legal():
        m[cm[c]] = True
    return m


class NumpyMLP:
    """ReLU MLP forward pass from a list of (W, b) layers (W shape: out x in)."""

    def __init__(self, layers):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in layers]

    def __call__(self, x):
        h = x
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b if h.ndim == 1 else h @ W.T + b
            if i < last:
                h = np.maximum(h, 0.0)
        return h

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        if z["W0"].shape[1] != FEATURE_DIM:
            raise ValueError(f"{path} expects {z['W0'].shape[1]} features, encoder has {FEATURE_DIM}; retrain")
        return cls([(z[f"W{i}"], z[f"b{i}"]) for i in range(n)])

    def save(self, path):
        d = {}
        for i, (W, b) in enumerate(self.layers):
            d[f"W{i}"], d[f"b{i}"] = W, b
        np.savez(path, **d)


class RLPlayer(HeuristicAgent):
    """Plays cards greedily w.r.t. the learned Q-network (bids with the heuristic unless
    wrapped in a Bot with a MonteCarloBidder)."""

    def __init__(self, net, epsilon=0.0, rng=None):
        super().__init__()
        self.net = NumpyMLP.load(net) if isinstance(net, str) else net
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

    def q_values(self, state):
        """{card: Q} for each legal card for the acting seat."""
        q = self.net(encode(state))
        cm = CANON[state.trump]
        return {c: float(q[cm[c]]) for c in state.legal()}

    def play(self, state):
        legal = state.legal()
        if len(legal) == 1:
            return legal[0]
        if self.epsilon and self.rng.random() < self.epsilon:
            return legal[self.rng.integers(len(legal))]
        qs = self.q_values(state)
        return max(qs, key=qs.get)


def is_trump(c, t):
    return TRUMP_POWER[t][c] >= 0
