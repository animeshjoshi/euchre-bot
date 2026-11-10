const SU = "SCHD", RK = "9TJQKA", SYM = ["♠", "♣", "♥", "♦"], SNAME = ["Spades","Clubs","Hearts","Diamonds"];
const SEAT = ["You", "West", "Partner", "East"];
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let S = null, busy = false;

// theme
(function(){ try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch(e){} })();
$("themeBtn").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const nxt = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nxt; try { localStorage.setItem("theme", nxt); } catch(e){}
};
document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === "v-" + b.dataset.v));
});

function cardHTML(c, cls = "", attrs = "") {
  if (c === null || c === undefined) return "";
  const s = Math.floor(c / 6), r = RK[c % 6] === "T" ? "10" : RK[c % 6];
  const red = s >= 2 ? " red" : "";
  return `<div class="card${red} ${cls}" ${attrs}><div class="r">${r}<small>${SYM[s]}</small></div><div class="big">${SYM[s]}</div></div>`;
}
const cstr = c => (RK[c % 6] === "T" ? "10" : RK[c % 6]) + SYM[Math.floor(c / 6)];

// ------------------------------------------------------------ the bot (Python in a worker)
// The game and the bots run in worker.js, so the page only draws what the worker says.
const worker = new Worker("worker.js");
const calls = new Map();
let nextCall = 1;
const boot = {text: "Starting Python…", failed: false};
function showBoot() { $("boot").hidden = boot.done; $("boot").className = "boot" + (boot.failed ? " err" : ""); $("bootText").textContent = boot.text; }
worker.onmessage = ({data: m}) => {
  if (m.type === "status") { boot.text = m.text; showBoot(); }
  else if (m.type === "ready") { boot.done = true; showBoot(); }
  else if (m.type === "fatal") { boot.failed = true; boot.text = "Could not start the bot: " + m.error; showBoot(); }
  else { const c = calls.get(m.id); calls.delete(m.id); m.out.ok ? c.resolve(m.out) : c.reject(new Error(m.out.error)); }
};
worker.onerror = e => { boot.failed = true; boot.text = "Could not start the bot: " + (e.message || "worker failed"); showBoot(); };
function call(op, arg) {
  return new Promise((resolve, reject) => { const id = nextCall++; calls.set(id, {resolve, reject}); worker.postMessage({id, op, arg}); });
}
function fail(e) { console.warn(e); boot.done = false; boot.failed = true; boot.text = "Something went wrong: " + e.message + ". Press New game to start again."; showBoot(); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

// The worker never waits for us: each step says what to draw and how long to pause before the next.
// wait_ms === null means it's the human's turn, so we stop until they act.
let token = 0;  // bumped by every new game so an old game's loop stops
async function advance(out, mine) {
  for (;;) {
    if (mine !== token) return;  // a newer game started while we were waiting
    S = out.state; render();
    if (out.wait_ms === null) return;
    if (out.wait_ms > 0) await sleep(out.wait_ms);
    if (mine !== token) return;
    out = await call("step");
    if (mine !== token) return;
  }
}

function evTable(rows, recommend = true) {
  if (!rows || !rows.length) return "";
  const m = Math.max(1, ...rows.map(r => Math.abs(r.ev)));
  return `<table class="evt"><tr><th>Option</th><th class="num">EV</th><th></th><th class="num">P(win)</th></tr>` +
    rows.map((r, i) => {
      const w = Math.abs(r.ev) / m * 50, cls = r.ev >= 0 ? "pos" : "neg";
      const left = r.ev >= 0 ? 50 : 50 - w;
      return `<tr class="${i === 0 && recommend ? "top" : ""}"><td>${i === 0 && recommend ? "★ " : ""}${esc(r.label)}</td>
        <td class="num">${r.ev >= 0 ? "+" : ""}${r.ev.toFixed(2)}<span class="muted" style="font-weight:400"> ±${r.se.toFixed(2)}</span></td>
        <td><div class="bar"><i class="${cls}" style="left:${left}%;width:${w}%"></i><span class="zero"></span></div></td>
        <td class="num">${r.p === null ? "" : (r.p * 100).toFixed(0) + "%"}</td></tr>`;
    }).join("") + `</table>`;
}

// ------------------------------------------------------------ play view
function render() {
  if (!S || S.phase === "none") return;
  const P = S.pending, hints = $("hintChk").checked;
  $("score").innerHTML = `<span class="pill">You &amp; Partner <b>${S.score[0]}</b></span><span class="pill">Bots <b>${S.score[1]}</b></span>`;
  const C = S.contract;
  $("info").innerHTML = [
    ["Hand", S.hand_no], ["Dealer", SEAT[S.dealer]],
    ["Trump", C ? `<b style="color:${C.trump >= 2 ? "var(--red)" : "inherit"}">${SYM[C.trump]} ${C.trump_name}</b>` : "—"],
    ["Maker", C ? SEAT[C.maker] + (C.alone ? " (alone)" : "") : "—"],
    ["Tricks", `You/Partner ${S.tricks_won[0]} – ${S.tricks_won[1]} West/East`],
    ["Card play", S.has_model ? "RL model" : "heuristic (no model yet)"], ["Game to", S.target],
  ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");

  // seats
  for (const p of [1, 2, 3]) {
    const bid = S.bids[p], thinking = S.thinking === p;
    const n = S.counts[p], vert = p !== 2;
    const out = C && C.alone && (C.maker + 2) % 4 === p;
    $("seat" + p).innerHTML = `<div class="name ${S.turn === p ? "active" : ""}">${SEAT[p]}${S.dealer === p ? '<span class="tag">Dealer</span>' : ""}</div>
      ${out ? '<div class="thinking">sitting out</div>' : `<div class="backs ${vert ? "v" : ""}">${'<div class="back"></div>'.repeat(n)}</div>`}
      ${bid ? `<div class="bubble">${esc(bid)}</div>` : ""}${thinking ? '<div class="thinking">thinking…</div>' : ""}`;
  }
  // trick slots
  for (let p = 0; p < 4; p++) $("slot" + p).innerHTML = "";
  for (const t of S.trick) $("slot" + t.seat).innerHTML = cardHTML(t.card);
  for (let p = 0; p < 4; p++) $("slot" + p).classList.toggle("win", S.trick_winner === p);
  // middle
  let mid = "";
  if (S.phase === "bidding" || (P && P.type === "discard")) {
    const st = S.upcard_state;
    mid = `<div class="lbl">${st === "up" ? "Upcard" : "Turned down"} · ${SEAT[S.dealer]} dealt</div>` +
      cardHTML(S.upcard, st === "up" ? "" : "dim") + `<div class="lbl">Bidding round ${S.bid_round}</div>`;
  } else if (S.phase === "play" && C) {
    mid = `<div class="lbl">Trump</div><div style="font-size:34px;color:${C.trump >= 2 ? "#ffb4ab" : "var(--felt-ink)"}">${SYM[C.trump]}</div>`;
    if (!S.trick.length && S.last_trick) mid += `<div class="lbl">Last trick: ${SEAT[S.last_trick.winner]}</div>`;
  } else if (S.result) {
    mid = `<div class="result">${esc(S.result)}</div>`;
  }
  $("mid").innerHTML = mid;

  // my seat: hand + actions
  const legal = P && P.type === "play" ? new Set(P.legal) : null;
  const discard = P && P.type === "discard" ? new Set(P.cards) : null;
  let best = null;
  if (legal && hints && P.q && Object.keys(P.q).length > 1) best = +Object.entries(P.q).sort((a, b) => b[1] - a[1])[0][0];
  if (discard && hints && P.best !== undefined) best = P.best;
  const hand = S.hand.map(c => {
    let cls = "", q = "";
    if (legal) { cls = legal.has(c) ? "legal" : "illegal"; if (hints && legal.has(c) && P.q[c] !== undefined && P.q_is_model) q = `<div class="qv">${P.q[c] >= 0 ? "+" : ""}${P.q[c].toFixed(2)}</div>`; }
    if (discard) cls = "legal";
    if (best === c) cls += " best";
    return `<div class="wrap">${q}${cardHTML(c, cls, `data-c="${c}"`)}</div>`;
  }).join("");
  let acts = "";
  if (P) {
    if (P.type === "bid1") {
      acts = `<button class="btn" data-a='{"kind":"pass"}'>Pass</button>
        <button class="btn" data-a='{"kind":"order"}'>${P.dealer_is_you ? "Pick up" : "Order up"} ${SYM[P.suit]}</button>
        <button class="btn" data-a='{"kind":"order","alone":true}'>${P.dealer_is_you ? "Pick up" : "Order up"} ${SYM[P.suit]} alone</button>`;
    } else if (P.type === "bid2") {
      acts = (P.must_call ? "" : `<button class="btn" data-a='{"suit":null}'>Pass</button>`) +
        P.suits.map(s => `<button class="btn" data-a='{"suit":${s}}'>${SYM[s]} ${SNAME[s]}</button>`).join("") +
        `<label><input type="checkbox" id="aloneChk"> alone</label>`;
    } else if (P.type === "discard") acts = `<span>Click a card to discard</span>`;
    else if (P.type === "play") acts = `<span>${S.trick.length ? "Your play" : "Your lead"} — click a card</span>`;
    else if (P.type === "next") acts = (S.result ? `<span class="result">${esc(S.result)}</span>` : "") + `<button class="btn primary" data-a='{}'>Next hand</button>`;
    else if (P.type === "new_game") acts = `<span class="result">${esc(S.result)}</span><button class="btn primary" id="again">New game</button>`;
  } else if (S.thinking === 0) acts = `<span>Simulating…</span>`;
  $("seat0").innerHTML = `<div class="name ${S.turn === 0 ? "active" : ""}">You${S.dealer === 0 ? '<span class="tag">Dealer</span>' : ""}
      ${S.bids[0] ? `<span class="bubble">${esc(S.bids[0])}</span>` : ""}</div>
    <div class="hand">${hand}</div><div class="actions">${acts}</div>`;

  // reveal at hand end
  if (S.reveal && S.phase !== "play") {
    $("seat0").innerHTML += `<div class="panel" style="margin-top:12px;color:var(--ink);width:100%;max-width:620px">
      <h3>Hands this deal (after pick-up)</h3>${[1, 2, 3].map(p => `<div><b>${SEAT[p]}:</b> ${S.reveal[p].map(c => `<span style="color:${c >= 12 ? "var(--red)" : "inherit"}">${cstr(c)}</span>`).join(" ")}</div>`).join("")}</div>`;
  }

  // hint panel
  let h = '<span class="muted">Advice appears when it’s your turn.</span>';
  if (!hints) h = '<span class="muted">Hidden.</span>';
  else if (P && (P.type === "bid1" || P.type === "bid2")) h = `<div class="muted" style="margin-bottom:6px">Monte Carlo over deals of the unseen cards (signed points for your team):</div>` + evTable(P.analysis);
  else if (P && P.type === "discard") h = `<div class="muted" style="margin-bottom:6px">Expected points by discard:</div>` + evTable(P.analysis);
  else if (P && P.type === "play") {
    const rows = Object.entries(P.q).sort((a, b) => b[1] - a[1]);
    h = P.q_is_model
      ? `<div class="muted" style="margin-bottom:6px">RL Q-values: expected hand points if you play each card.</div><table class="evt">${rows.map(([c, q], i) => `<tr class="${i ? "" : "top"}"><td>${i ? "" : "★ "}${cstr(+c)}</td><td class="num">${q >= 0 ? "+" : ""}${q.toFixed(3)}</td></tr>`).join("")}</table>`
      : `<div>Heuristic suggests <b>${cstr(+rows[0][0])}</b> (train a model for Q-values).</div>`;
  }
  $("hint").innerHTML = h;
  $("log").innerHTML = S.log.map(l => `<div>${esc(l)}</div>`).join("");
  $("log").scrollTop = 1e6;
}

async function act(a) {
  if (busy) return; busy = true;
  const mine = token;
  try {
    try { await call("act", a); }
    catch (e) { console.warn(e); return; }  // the worker refused the answer; the question is still open
    if (mine !== token) return;             // "New game" was pressed meanwhile: don't step the new game
    await advance(await call("step"), mine);
  } catch (e) { if (mine === token) fail(e); } finally { busy = false; }
}
$("table").addEventListener("click", e => {
  const P = S && S.pending; if (!P) return;
  const b = e.target.closest("[data-a]");
  if (b) { const a = JSON.parse(b.dataset.a); const al = $("aloneChk"); if (P.type === "bid2" && al) a.alone = al.checked; return act(a); }
  if (e.target.closest("#again")) return newGame();
  const c = e.target.closest(".card[data-c]");
  if (c && (P.type === "play" || P.type === "discard") && c.classList.contains("legal")) act({card: +c.dataset.c});
});
$("hintChk").onchange = render;
async function newGame() {
  const mine = ++token;
  S = null; busy = false;  // ignore clicks on the old table until the new game arrives
  try {
    const out = await call("new_game", {stick: $("stickChk").checked});
    if (mine === token) { boot.failed = false; boot.done = true; showBoot(); await advance(out, mine); }
  } catch (e) { if (mine === token) fail(e); }
}
$("newBtn").onclick = newGame;
newGame();

// ------------------------------------------------------------ analyzer
const A = {hand: [], up: null, mode: "hand", dealer: 3, round: 1};
function renderPicker() {
  let h = "";
  for (const s of [0, 2, 1, 3]) for (let r = 0; r < 6; r++) {
    const c = s * 6 + r;
    h += cardHTML(c, A.hand.includes(c) ? "inhand" : A.up === c ? "isup" : "", `data-c="${c}"`);
  }
  $("picker").innerHTML = h;
  $("selHand").innerHTML = A.hand.length ? A.hand.map(c => cardHTML(c)).join("") : '<span class="muted">none</span>';
  $("selUp").innerHTML = A.up !== null ? cardHTML(A.up, A.round === 2 ? "dim" : "") : '<span class="muted">none</span>';
}
$("picker").onclick = e => {
  const el = e.target.closest(".card[data-c]"); if (!el) return; const c = +el.dataset.c;
  if (A.mode === "hand") {
    if (A.hand.includes(c)) A.hand = A.hand.filter(x => x !== c);
    else if (A.hand.length < 5) { if (A.up === c) A.up = null; A.hand.push(c); if (A.hand.length === 5 && A.up === null) setMode("up"); }
  } else { A.up = A.up === c ? null : c; A.hand = A.hand.filter(x => x !== c); }
  renderPicker();
};
function setMode(m) { A.mode = m; document.querySelectorAll("#pickMode button").forEach(b => b.classList.toggle("on", b.dataset.m === m)); }
document.querySelectorAll("#pickMode button").forEach(b => b.onclick = () => setMode(b.dataset.m));
document.querySelectorAll("#dealerSeg button").forEach(b => b.onclick = () => { A.dealer = +b.dataset.d; document.querySelectorAll("#dealerSeg button").forEach(x => x.classList.toggle("on", x === b)); });
document.querySelectorAll("#roundSeg button").forEach(b => b.onclick = () => { A.round = +b.dataset.r; document.querySelectorAll("#roundSeg button").forEach(x => x.classList.toggle("on", x === b)); renderPicker(); });
$("sims").oninput = () => $("simsLbl").textContent = $("sims").value;
$("clrBtn").onclick = () => { A.hand = []; A.up = null; setMode("hand"); renderPicker(); };
$("randBtn").onclick = () => {
  const d = [...Array(24).keys()].sort(() => Math.random() - .5);
  A.hand = d.slice(0, 5); A.up = d[5]; renderPicker();
};
$("anBtn").onclick = async () => {
  if (A.hand.length !== 5 || A.up === null) { $("anOut").innerHTML = '<span class="err">Pick exactly 5 hand cards and an upcard.</span>'; return; }
  $("anBtn").disabled = true; $("anOut").innerHTML = `<span class="muted">Simulating ${$("sims").value} deals…</span>`;
  try {
    const r = (await call("analyze", {hand: A.hand, upcard: A.up, dealer: A.dealer, round: A.round, sims: +$("sims").value, stick: $("anStick").checked})).analysis;
    const pos = (0 - A.dealer - 1 + 8) % 4;
    $("anOut").innerHTML = `<div style="margin-bottom:8px"><b>Best: ${esc(r.results[0].label)}</b>
      <div class="muted">Round ${A.round}, you bid ${["1st", "2nd", "3rd", "4th"][pos]} · ${r.sims} deals in ${r.seconds}s · rollouts: ${r.rollout}</div></div>` + evTable(r.results);
  } catch (e) { $("anOut").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  $("anBtn").disabled = false;
};
renderPicker();
