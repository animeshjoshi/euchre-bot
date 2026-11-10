"""The functions the browser calls (through Pyodide in a Web Worker).

Everything goes in and out as JSON text, so the JavaScript side never has to
handle Python objects. Each function returns {"ok": true, ...} or
{"ok": false, "error": "..."}.
"""
import json

from .webgame import Game, analyze_bid, load_player

_s = {"player": None, "has_model": False, "game": None}


def _reply(fn):
    def wrapper(*args):
        try:
            return json.dumps({"ok": True, **fn(*args)})
        except Exception as e:  # report to the page instead of dying inside the worker
            return json.dumps({"ok": False, "error": str(e) or type(e).__name__})
    return wrapper


@_reply
def init(model_path):
    _s["player"], _s["has_model"] = load_player(model_path)
    return {"has_model": _s["has_model"]}


@_reply
def new_game(opts_json):
    opts = json.loads(opts_json)
    _s["game"] = Game(_s["player"], _s["has_model"], sims=int(opts.get("sims", 200)),
                      delay_ms=int(opts.get("delay_ms", 800)), target=int(opts.get("target", 10)),
                      stick=bool(opts.get("stick", False)), seed=opts.get("seed"))
    return _s["game"].step()


@_reply
def step():
    if _s["game"] is None:
        raise RuntimeError("no game in progress")
    return _s["game"].step()


@_reply
def act(answer_json):
    if _s["game"] is None:
        raise RuntimeError("no game in progress")
    err = _s["game"].submit(json.loads(answer_json))
    if err:
        raise ValueError(err)
    return {}


@_reply
def analyze(req_json):
    return {"analysis": analyze_bid(json.loads(req_json), _s["player"], _s["has_model"])}
