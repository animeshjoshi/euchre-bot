// Smoke test for the part the Python tests can't cover: does the built site's code
// run under Pyodide (Python in WebAssembly)? Plays a short game and runs the analyzer,
// using the same boot steps as web/worker.js.
//
//   python scripts/build_site.py && npm install --no-save pyodide@<version in worker.js>
//   node tests/pyodide_smoke.mjs dist
import fs from "node:fs";
import { loadPyodide } from "pyodide";

const dist = process.argv[2] || "dist";
const lock = JSON.parse(fs.readFileSync("node_modules/pyodide/pyodide-lock.json", "utf8"));
const version = /v(\d+\.\d+\.\d+)\/full/.exec(fs.readFileSync(`${dist}/worker.js`, "utf8"))[1];
const bytes = path => new Uint8Array(fs.readFileSync(path));

const py = await loadPyodide();
await py.loadPackage(`https://cdn.jsdelivr.net/pyodide/v${version}/full/${lock.packages.numpy.file_name}`);
py.unpackArchive(bytes(`${dist}/py/euchre.zip`), "zip", { extractDir: "/app" });
py.runPython("import sys; sys.path.insert(0, '/app')");
py.FS.mkdirTree("/models");
py.FS.writeFile("/models/euchre_q.npz", bytes(`${dist}/py/euchre_q.npz`));
const bridge = py.pyimport("euchre.webbridge");
const call = (fn, arg) => {
  const out = JSON.parse(arg === undefined ? bridge[fn]() : bridge[fn](JSON.stringify(arg)));
  if (!out.ok) throw new Error(`${fn}: ${out.error}`);
  return out;
};

const init = JSON.parse(bridge.init("/models/euchre_q.npz"));
if (!init.ok || !init.has_model) throw new Error("model did not load: " + JSON.stringify(init));

let out = call("new_game", { target: 2, delay_ms: 0, seed: 11 });
let decisions = 0;
for (let i = 0; i < 20000; i++) {
  out = call("step");
  const p = out.state.pending;
  if (!p) continue;
  if (p.type === "new_game") break;
  call("act", p.type === "play" ? { card: p.legal[0] }
    : p.type === "discard" ? { card: p.cards[0] }
    : p.type === "bid1" ? { kind: "pass" }
    : p.type === "bid2" ? { suit: p.must_call ? p.suits[0] : null } : {});
  decisions++;
}
if (out.state.phase !== "game_over") throw new Error("game did not finish: " + out.state.phase);

const an = call("analyze", { hand: [0, 1, 2, 3, 4], upcard: 5, dealer: 3, round: 1, sims: 100 }).analysis;
if (an.rollout !== "RL model" || an.results.length < 2) throw new Error("analyzer gave an odd result");
console.log(`ok: Pyodide ${version}, game finished after ${decisions} decisions (score ${out.state.score}), analyzer best: ${an.results[0].label}`);
