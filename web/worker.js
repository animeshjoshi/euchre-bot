// Runs the Euchre bot in the browser. Python (Pyodide) lives here, off the page's
// main thread, so the bots' Monte Carlo thinking never freezes the UI.
"use strict";

// Pinned on purpose: a new Pyodide release should never change the site by surprise.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const MODEL_PATH = "/models/euchre_q.npz";
const BUILD = "__BUILD__";  // scripts/build_site.py swaps in a hash of the files below, so a redeploy skips stale caches
const OPS = new Set(["new_game", "step", "act", "analyze"]);

importScripts(PYODIDE_URL + "pyodide.js");

const status = text => postMessage({type: "status", text});

async function fetchBytes(url, required) {
  const r = await fetch(url);
  if (!r.ok) {
    if (required) throw new Error(`could not load ${url} (${r.status})`);
    return null;
  }
  return new Uint8Array(await r.arrayBuffer());
}

async function boot() {
  status("Starting Python…");
  const py = await loadPyodide({indexURL: PYODIDE_URL});
  status("Loading numpy…");
  await py.loadPackage("numpy");
  status("Loading the bot…");
  const [code, model] = await Promise.all([fetchBytes(`py/euchre.zip?v=${BUILD}`, true), fetchBytes(`py/euchre_q.npz?v=${BUILD}`, false)]);
  py.unpackArchive(code, "zip", {extractDir: "/app"});
  py.runPython("import sys; sys.path.insert(0, '/app')");
  if (model) {  // without it the bots fall back to rule-based card play
    py.FS.mkdirTree("/models");
    py.FS.writeFile(MODEL_PATH, model);
  }
  const bridge = py.pyimport("euchre.webbridge");
  const init = JSON.parse(bridge.init(MODEL_PATH));
  if (!init.ok) throw new Error(init.error);
  return bridge;
}

const ready = boot();
ready.then(() => postMessage({type: "ready"}), e => postMessage({type: "fatal", error: String(e)}));

onmessage = async ({data: {id, op, arg}}) => {
  try {
    if (!OPS.has(op)) throw new Error(`unknown op: ${op}`);
    const bridge = await ready;
    const out = JSON.parse(arg === undefined ? bridge[op]() : bridge[op](JSON.stringify(arg)));
    postMessage({id, out});
  } catch (e) {
    postMessage({id, out: {ok: false, error: String(e && e.message || e)}});
  }
};
