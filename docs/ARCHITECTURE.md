# Architecture

The bot runs entirely in the visitor's browser, so the site can be hosted on GitHub Pages, which only serves static files. There is no server.

## How it fits together

```
 GitHub                          Visitor's browser
 ──────                          ─────────────────────────────────────────────
 push to main                    index.html + app.js        (main thread)
      │                            draws the table, handles clicks
      ▼                                   │  messages: new_game / step / act / analyze
 GitHub Actions                           ▼
   1. unit tests                  worker.js                 (Web Worker)
   2. build dist/                   loads Pyodide + numpy, then calls
   3. Pyodide smoke test                  │
   4. deploy to Pages                     ▼
      │                           euchre/webbridge.py  ──►  euchre/webgame.py
      ▼                             JSON in, JSON out        the game, one step at a time
 https://<user>.github.io/<repo>/                                  │
                                                                   ▼
                                   euchre/engine.py, mc_bidder.py, rl.py ...
                                   (the same bot code the command-line tools use)
```

## Why Pyodide

Pyodide is Python and numpy compiled to WebAssembly. The bot is about 1,400 lines of Python plus a numpy neural network, so running it as-is means one copy of the rules and the bot, with nothing rewritten in JavaScript to drift out of sync. The cost is a first-visit download (about 10 MB, cached afterwards) and speed: the same game runs about 5x slower than native Python.

## Why a Web Worker

The bots think by simulating hundreds of deals, which takes seconds. On the page's main thread that would freeze the screen. In a worker the page stays responsive and shows "thinking…".

## The game loop (the part that changed most)

The old server ran the game in a thread that stopped and waited for the human's click. Browsers' Python can't run threads, so `webgame.Game` is a step-by-step state machine:

- `step()` runs the game up to the next moment worth showing and returns `{state, wait_ms}`.
  - `wait_ms` is a number: draw the state, wait that long (so bot moves are visible), then call `step()` again.
  - `wait_ms` is `null`: it is the human's turn. `state.pending` says what is being asked.
- `submit(answer)` gives the human's answer. It rejects anything malformed or illegal and leaves the question open.
- A bot's "thinking" state is returned first (with a wait of 0) and its decision on the next step, so the page can draw "thinking…" before the slow part starts.

Bidding uses `engine.iter_bidding`, a generator that yields each bid or discard request. `engine.run_bidding` drives the same generator with one bot per seat, so the command-line tools and the web game share one set of bidding rules.

## Files

| Path | Role |
|---|---|
| `euchre/webgame.py` | The game as a state machine (no threads, no sleeping, no HTTP) |
| `euchre/webbridge.py` | The few functions the page calls. JSON in, JSON out; errors come back as `{ok: false, error}` |
| `web/index.html`, `web/app.js` | The page: drawing and clicks |
| `web/worker.js` | Starts Pyodide, loads the code and model, answers messages |
| `scripts/build_site.py` | Builds `dist/`: the web files, `py/euchre.zip` and `py/euchre_q.npz` |
| `tests/` | Python unit tests, plus `pyodide_smoke.mjs` that runs the built site's code in real Pyodide |
| `.github/workflows/pages.yml` | Test, build and deploy |

## Deploying

One-time setup, in the GitHub repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
After that, every push to `main` that passes the tests is published. Pull requests run the same checks and are not published.

Run it locally (no install beyond numpy; needs internet for the Pyodide CDN):

```bash
python scripts/build_site.py
python -m http.server -d dist 8000     # open http://localhost:8000
```

## Decisions and trade-offs

- **Pyodide is pinned** (`web/worker.js`). Upgrading is a one-line change, and CI runs the same version in its smoke test, so an upgrade that breaks the bot fails the build instead of the live site.
- **Pyodide loads from the jsDelivr CDN** rather than being copied into the repo, which keeps the repo small. If the CDN is down the page shows an error banner.
- **Cache-busting.** The worker requests `py/euchre.zip?v=<hash>` and the model the same way, with a hash of their contents stamped in at build time, so a new deploy never pairs new code with an old cached model.
- **Everything is per visitor.** Each browser has its own game, so any number of people can play at once and nothing is stored.
- **Speed.** Expect a bot decision in 1-5 seconds and the "Bot advice" analysis in up to about 5 seconds. Lowering the simulation counts in `webgame.Game` (`sims`, `hint_sims`) makes it faster and the bots slightly weaker.
