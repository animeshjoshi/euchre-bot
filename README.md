# Euchre bot

A Euchre bot written in Python that makes two kinds of decision:

1. **Trump calling (Monte Carlo).** It decides whether to order up, call a suit, go alone or pass. It knows only its own 5 cards, the upcard, and that the players before it passed.
2. **Card play (reinforcement learning).** A Q-network picks the card with the highest expected points.

It comes with a web UI where you can play against the bot or have it analyse a bidding decision, plus command-line tools and benchmarks.

Rules used: 24-card deck, two bidding rounds, dealer picks up and discards, going alone, and standard scoring (makers score 1 / 2 / 4, a euchre scores 2). Defending alone is not supported. Stick-the-dealer is optional.

## Setup

You need Python 3.10 or newer.

```bash
git clone https://github.com/animeshjoshi/euchre-bot.git
cd euchre-bot
pip install -r requirements.txt            # numpy only: enough to play and analyse
pip install -r requirements-train.txt      # adds torch, needed only to retrain the model
```

The trained model (`models/euchre_q.npz`) is included, so you don't need to train anything to start.

## How to use

### 1. Play in the browser

```bash
python server.py
```

Open **http://localhost:8000**.

- **Play tab:** you sit South. North is your bot partner, and West and East are bot opponents. The game goes to 10 points.
  - Bidding: use the buttons (Pass, Order up, Order up alone; in round 2, a suit, with an "alone" box).
  - Discarding and playing: click a card.
  - **Bot advice** (in the right-hand panel) shows what the bot would do in your seat:
    - for a bid or discard, a table of the expected points for each option;
    - while playing, the RL network's expected points above each card you could play, with the best card outlined.
- **Bid analyzer tab:** click 5 cards for your hand, switch to *Upcard* and click the upcard, then choose the dealer and the round. Press **Analyze** to get the expected points for every option.

### 2. Analyse a bid from the command line

```bash
python demo.py bid --hand "JD AH KH 9C AS" --upcard 9H --dealer partner
```

```
  order up H                       EV +1.163 +/- 0.021   P(win hand) 97.3%
  order up H ALONE                 EV +0.829 +/- 0.024   P(win hand) 93.8%
  pass                             EV -0.106 +/- 0.034   P(win hand) 19.5%
=> best: order up H
```

- **Cards** are written as rank then suit. Ranks are `9 T J Q K A` (`10` also works). Suits are `S C H D`. For example, `JD` is the jack of diamonds.
- **`--dealer`** says who dealt, relative to you: `me`, `left`, `partner` or `right`.
- **`--round 2`** means the upcard was turned down, so you choose another suit.
- **`--stick`** turns on the stick-the-dealer rule.
- **`--sims N`** sets the number of simulated deals (default 1000).
- **EV** is the expected points per hand for your team: + when you score, − when the opponents do.

### 3. Watch the bot play a whole hand

```bash
python demo.py hand --seed 7
```

This prints every seat's bidding analysis, then each card played, along with the network's Q-values.

### 4. Benchmark

```bash
python benchmark.py
```

This plays duplicate matches against rule-based and random players (see Results).

### 5. Retrain the card player (optional)

```bash
python train_rollout.py --iters 6 --states 100000 --k 16 --out models/euchre_q.npz
```

This takes about 45 minutes on a 14-core CPU. The network is evaluated against the heuristic after every iteration, and the best version is saved.

### 6. Use it from code

```python
from euchre.engine import Bot, BidView
from euchre.mc_bidder import MonteCarloBidder
from euchre.rl import RLPlayer
from euchre.cards import parse_cards, parse_card

rl = RLPlayer("models/euchre_q.npz")
bot = Bot(MonteCarloBidder(n_sims=500, play_policy=rl), rl)      # an engine.Agent

# seat 0, dealer is seat 3 (right of you), you bid 2nd in round 1
v = BidView(player=0, hand=parse_cards("JH JD AH 9C KS"), upcard=parse_card("QH"),
            dealer=3, round=1, position=1)
for r in bot.bidder.analyze(v):
    print(r)
# during play: rl.q_values(play_state) -> {card: expected points}; rl.play(play_state) -> best card
```

## Project layout

| Path | What it is |
|---|---|
| `euchre/cards.py` | Card encoding and trump tables (bowers, effective suit) |
| `euchre/engine.py` | Rules: deal, two-round bidding, pick-up/discard, trick play, scoring |
| `euchre/heuristics.py` | Rule-based bidder and player (baseline and opponent model) |
| `euchre/mc_bidder.py` | Monte Carlo trump-call / pass decision |
| `euchre/rl.py` | State encoder and numpy Q-network player |
| `euchre/search.py` | Play-time search over sampled hidden deals |
| `euchre/evaluate.py` | Duplicate-match evaluation |
| `train_rollout.py` | RL trainer used for the shipped model (rollout policy iteration) |
| `train.py` | Simpler self-play Monte Carlo Q-learning trainer (kept for comparison) |
| `server.py`, `web/index.html` | Web UI (Python standard library only) |
| `demo.py`, `benchmark.py` | Command-line tools |

## How the bidding works

For each decision, the bidder:

1. **Samples N worlds.** The 18 unseen cards are dealt at random to the other three hands and the 3 buried kitty cards. If a player who already passed would have bid with their sampled hand under the opponent model, that world is rejected and redrawn, so the bidder learns from the passes.
2. **Lists every legal action:** pass, order up, order up alone, and in round 2, call a suit or call it alone. When the bot is the dealer, each possible discard counts as a separate action.
3. **Plays every action to the end in every world,** with the RL network playing all four seats. A pass is valued by continuing the bidding with the opponent model. If everyone passes, the result is a redeal, worth 0.
4. **Picks the action with the highest mean signed points.** Every action is scored on the same worlds, so the comparison between actions has low variance.

## How the card play works

- **State (261 features):**
  - your hand, the current trick by seat, cards already played, the led suit, known voids;
  - who made trump and whether they went alone, tricks won so far, trick number, your position in the trick, who is currently winning it;
  - the upcard, whether the dealer took it, and who dealt;
  - card tracking: for each card in your hand, whether it is the highest card left in its suit, how many unseen cards beat it, and whether it beats the current winning card; plus unseen-card counts per suit.

  Suits are relabelled so that trump is always suit 0, so one network covers every trump suit.
- **Model:** an MLP mapping 261 → 256 → 256 → 24 Q-values. Illegal cards are masked out. Inference uses numpy only.
- **Training (rollout policy iteration):**
  1. Sample training positions.
  2. For each position, deal the hidden cards at random K times, consistent with everything the player has seen.
  3. In each deal, try every legal card and play the hand out with the current policy.
  4. The mean result for each card becomes its Q-target.
  5. Fit the network to the targets. The fitted network becomes the play-out policy for the next iteration.

  An earlier version learned from final hand scores through self-play (`train.py`). It plateaued below the heuristic because those targets are too noisy.

## Results

These are duplicate matches: every deal is played twice with the teams swapping seats, which cancels most card luck. Each number is the first side's points per hand, ± one standard error (from `models/benchmark.txt`).

| Matchup | Points per hand |
|---|---|
| Card play: RL vs heuristic | **+0.030 ± 0.006** (4000 deals) |
| Card play: RL vs random | +0.333 ± 0.009 |
| Card play: heuristic vs random | +0.320 ± 0.009 |
| Bidding: Monte Carlo vs heuristic (both use RL play) | **+0.233 ± 0.057** (200 deals) |
| Full bot (MC bidding + RL play) vs heuristic bot | **+0.225 ± 0.057** (200 deals) |

Most of the edge comes from bidding. Card play leaves less room for gains: even a random player loses only about 0.32 points per hand to the heuristic. Play-time search scores about +0.03 against the heuristic, the same as the RL network, which picks its card instantly.

## Limitations

- In the "pass" rollouts, the other players' bids, and the bot's own later bids, are modelled by the heuristic bidder.
- The bot maximises points for the current hand and ignores the game score (for example at 9–9).
- Defending alone is not supported.
