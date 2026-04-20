// ASL translator — MediaPipe HandLandmarker + rule-based letter classifier.
// Runs entirely in the browser. No server, no tracking, no uploads.

import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const letterEl = document.getElementById("letter");
const confBar = document.getElementById("confBar");
const confText = document.getElementById("confText");
const bufferEl = document.getElementById("buffer");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const speakBtn = document.getElementById("speak");
const spaceBtn = document.getElementById("space");
const backBtn = document.getElementById("back");
const clearBtn = document.getElementById("clear");
const autoEl = document.getElementById("auto");

let landmarker = null;
let stream = null;
let rafId = null;
let drawer = null;
let lastVideoTime = -1;

// auto-append tracking
const HOLD_MS = 1000;
let lastLetter = null;
let letterStart = 0;
let lastAppended = null;

// Temporal smoother. The raw classifier runs per frame (~30 fps) and a brief
// mis-read can flicker between poses during transitions. Keep the last N raw
// predictions and report the majority vote — this steadies the displayed letter
// and stops transient noise from triggering auto-append.
const SMOOTH_WINDOW = 7;
const recent = [];

function setStatus(t) { statusEl.textContent = t; }

async function initLandmarker() {
  setStatus("Loading model…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  drawer = new DrawingUtils(ctx);
  setStatus("Model ready");
}

async function start() {
  startBtn.disabled = true;
  try {
    if (!landmarker) await initLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise((r) => (video.onloadedmetadata = r));
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    stopBtn.disabled = false;
    setStatus("Running");
    loop();
  } catch (e) {
    setStatus("Error: " + e.message);
    startBtn.disabled = false;
  }
}

function stop() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped");
  setLetter("—", 0);
  recent.length = 0;
  lastLetter = null;
  lastAppended = null;
}

function loop() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const t = performance.now();
    const res = landmarker.detectForVideo(video, t);
    drawFrame(res);
    handleResult(res, t);
  }
  rafId = requestAnimationFrame(loop);
}

function drawFrame(res) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!res.landmarks || res.landmarks.length === 0) return;
  for (const lm of res.landmarks) {
    drawer.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#6aa8ff", lineWidth: 3 });
    drawer.drawLandmarks(lm, { color: "#ffffff", lineWidth: 1, radius: 3 });
  }
}

function handleResult(res, t) {
  if (!res.landmarks || res.landmarks.length === 0) {
    pushSample({ letter: "—", confidence: 0 });
    lastLetter = null;
    setLetter("—", 0);
    return;
  }
  const hand = res.landmarks[0];
  const handedness = res.handedness?.[0]?.[0]?.categoryName ?? "Right";
  pushSample(classify(hand, handedness));
  const { letter, confidence } = smoothed();
  setLetter(letter, confidence);

  if (!autoEl.checked) return;
  if (letter === "?" || letter === "—" || confidence < 0.6) {
    lastLetter = null;
    return;
  }
  if (letter !== lastLetter) {
    lastLetter = letter;
    letterStart = t;
    lastAppended = null;
  } else if (letter !== lastAppended && t - letterStart >= HOLD_MS) {
    append(letter);
    lastAppended = letter;
  }
}

function pushSample(s) {
  recent.push(s);
  if (recent.length > SMOOTH_WINDOW) recent.shift();
}

function smoothed() {
  if (recent.length === 0) return { letter: "—", confidence: 0 };
  // Pick the letter that appears most often in the window; confidence is the
  // mean confidence of samples agreeing with the winner.
  const counts = new Map();
  for (const s of recent) counts.set(s.letter, (counts.get(s.letter) || 0) + 1);
  let best = "—", bestN = 0;
  for (const [k, n] of counts) if (n > bestN) { best = k; bestN = n; }
  const agree = recent.filter((s) => s.letter === best);
  const conf = agree.reduce((a, s) => a + s.confidence, 0) / agree.length;
  // Require a clear majority — otherwise the user is mid-transition.
  if (bestN * 2 <= recent.length) return { letter: "?", confidence: 0.3 };
  return { letter: best, confidence: conf };
}

function setLetter(l, c) {
  letterEl.textContent = l;
  confBar.style.width = Math.max(0, Math.min(1, c)) * 100 + "%";
  confText.textContent = l === "—" || l === "?" ? "" : `confidence ${(c * 100).toFixed(0)}%`;
}

function append(s) { bufferEl.textContent += s; renderSuggestions(); }

speakBtn.addEventListener("click", () => {
  const text = bufferEl.textContent.trim();
  if (!text) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
});
spaceBtn.addEventListener("click", () => append(" "));
backBtn.addEventListener("click", () => { bufferEl.textContent = bufferEl.textContent.slice(0, -1); renderSuggestions(); });
clearBtn.addEventListener("click", () => { bufferEl.textContent = ""; renderSuggestions(); });

// ————— word suggestions —————
// Load a ~5k-word frequency-sorted dictionary once; fingerspelling one letter
// at a time is slow, so showing completion chips for the current prefix makes
// the app actually usable for more than isolated-letter demos.
const suggestionsEl = document.getElementById("suggestions");
let dictionary = [];

(async () => {
  try {
    const r = await fetch("data/words.json");
    dictionary = await r.json();
    renderSuggestions();
  } catch {
    suggestionsEl.innerHTML = "";
  }
})();

// Number keys 1–5 pick the corresponding suggestion without leaving the page.
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.isContentEditable) return;
  const n = Number(e.key);
  if (!Number.isInteger(n) || n < 1 || n > 5) return;
  const chips = suggestionsEl.querySelectorAll(".suggestion");
  if (chips[n - 1]) chips[n - 1].click();
});

function currentPrefix() {
  const text = bufferEl.textContent;
  const tail = text.split(/\s+/).pop() || "";
  return tail.toLowerCase();
}

function suggest(prefix, limit = 5) {
  if (!prefix || dictionary.length === 0) return [];
  const out = [];
  for (const w of dictionary) {
    if (w.startsWith(prefix)) {
      out.push(w);
      if (out.length >= limit) break;
    }
  }
  return out;
}

function renderSuggestions() {
  if (!suggestionsEl) return;
  const prefix = currentPrefix();
  if (!prefix) {
    suggestionsEl.innerHTML = dictionary.length === 0
      ? `<span class="suggestion-hint">Loading dictionary…</span>`
      : `<span class="suggestion-hint">Type a letter to see suggestions. 1–5 to pick.</span>`;
    return;
  }
  const hits = suggest(prefix, 5);
  if (hits.length === 0) {
    suggestionsEl.innerHTML = `<span class="suggestion-hint">No matches for "${escapeHtml(prefix)}"</span>`;
    return;
  }
  suggestionsEl.innerHTML = hits
    .map((w, i) => `<button type="button" class="suggestion" data-word="${escapeHtml(w)}"><span class="idx">${i + 1}</span>${escapeHtml(w)}</button>`)
    .join("");
  for (const btn of suggestionsEl.querySelectorAll(".suggestion")) {
    btn.addEventListener("click", () => acceptSuggestion(btn.dataset.word));
  }
}

function acceptSuggestion(word) {
  // Replace the trailing partial word with the completed one + a space,
  // so the user can immediately start the next word.
  const text = bufferEl.textContent;
  const idx = text.search(/\S+$/);
  bufferEl.textContent = (idx < 0 ? text : text.slice(0, idx)) + word + " ";
  renderSuggestions();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// ————— classifier —————
// MediaPipe landmarks:
//   0 wrist · 1-4 thumb · 5-8 index · 9-12 middle · 13-16 ring · 17-20 pinky

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function fingerExtended(lm, tip, pip, mcp) {
  // Finger is "extended" when:
  //   1. tip is farther from the wrist than pip (basic length gate), and
  //   2. the segments (mcp→pip) and (pip→tip) point in the same direction,
  //      i.e. the finger is roughly straight rather than curled back.
  // Using a dot product instead of a raw y comparison is robust to a rotated
  // or tilted hand — the previous y<mcp.y rule failed whenever the user held
  // the hand sideways.
  const w = lm[0];
  const farEnough = dist(lm[tip], w) > dist(lm[pip], w) * 1.03;
  const v1x = lm[pip].x - lm[mcp].x, v1y = lm[pip].y - lm[mcp].y;
  const v2x = lm[tip].x - lm[pip].x, v2y = lm[tip].y - lm[pip].y;
  const len1 = Math.hypot(v1x, v1y) + 1e-6;
  const len2 = Math.hypot(v2x, v2y) + 1e-6;
  const cosine = (v1x * v2x + v1y * v2y) / (len1 * len2);
  // cos > 0.3 ≈ segments within ~70° of each other; tuned empirically.
  return farEnough && cosine > 0.3;
}

function thumbExtended(lm, handedness) {
  // Thumb extends sideways. For a right hand mirrored in the video, thumb tip is
  // further from the palm center along +x (or -x for left hand).
  const tip = lm[4], ip = lm[3], mcp = lm[2];
  const sideways = Math.abs(tip.x - mcp.x) > Math.abs(tip.y - mcp.y);
  const beyond = dist(tip, lm[0]) > dist(ip, lm[0]) * 1.02;
  return sideways && beyond;
}

function classify(lm, handedness) {
  const t = thumbExtended(lm, handedness);
  const i = fingerExtended(lm, 8, 6, 5);
  const m = fingerExtended(lm, 12, 10, 9);
  const r = fingerExtended(lm, 16, 14, 13);
  const p = fingerExtended(lm, 20, 18, 17);

  // Pair-specific geometry used by letters that share a coarse shape.
  const indexMiddleGap = dist(lm[8], lm[12]);
  const palmSpan = dist(lm[5], lm[17]) + 1e-6;
  const gap = indexMiddleGap / palmSpan; // wide for V, narrow for U

  const thumbTouchIndex = dist(lm[4], lm[8]) < palmSpan * 0.35;
  const fingersCurledToThumb =
    dist(lm[8], lm[4]) < palmSpan * 0.55 &&
    dist(lm[12], lm[4]) < palmSpan * 0.55 &&
    dist(lm[16], lm[4]) < palmSpan * 0.65 &&
    dist(lm[20], lm[4]) < palmSpan * 0.75;

  // Table lookup on finger state → letter.
  // Pattern: [thumb, index, middle, ring, pinky]
  const rules = [
    // Open palm (greeting)
    { pat: [1, 1, 1, 1, 1], out: "Hello", conf: 0.75 },
    // B — flat palm with thumb across
    { pat: [0, 1, 1, 1, 1], out: "B", conf: 0.85 },
    // W — three fingers
    { pat: [0, 1, 1, 1, 0], out: "W", conf: 0.85 },
    // V / U — index + middle
    { pat: [0, 1, 1, 0, 0], out: gap > 0.45 ? "V" : "U", conf: 0.85 },
    // D — only index
    { pat: [0, 1, 0, 0, 0], out: "D", conf: 0.8 },
    // I — only pinky
    { pat: [0, 0, 0, 0, 1], out: "I", conf: 0.85 },
    // Y — thumb + pinky
    { pat: [1, 0, 0, 0, 1], out: "Y", conf: 0.85 },
    // L — thumb + index
    { pat: [1, 1, 0, 0, 0], out: "L", conf: 0.85 },
    // A — closed fist, thumb out to side
    { pat: [1, 0, 0, 0, 0], out: "A", conf: 0.75 },
    // E — closed fist, thumb in
    { pat: [0, 0, 0, 0, 0], out: "E", conf: 0.7 },
  ];

  const state = [t, i, m, r, p].map((x) => (x ? 1 : 0));

  // Shape-based overrides take priority over the finger-pattern table.
  if (t && i && thumbTouchIndex && m && r && p) return { letter: "F", confidence: 0.8 };
  if (fingersCurledToThumb && !i && !m && !r && !p) return { letter: "O", confidence: 0.75 };
  if (cShape(lm)) return { letter: "C", confidence: 0.7 };

  for (const rule of rules) {
    if (rule.pat.every((v, k) => v === state[k])) {
      return { letter: rule.out, confidence: rule.conf };
    }
  }
  return { letter: "?", confidence: 0.3 };
}

function cShape(lm) {
  // A C has all fingers curled partially with the thumb opposing them,
  // and the tips roughly equidistant from the wrist.
  const w = lm[0];
  const span = dist(lm[5], lm[17]) + 1e-6;
  const tips = [4, 8, 12, 16, 20].map((k) => dist(lm[k], w) / span);
  const mean = tips.reduce((a, b) => a + b, 0) / tips.length;
  const spread = Math.max(...tips) - Math.min(...tips);
  const thumbOpen = dist(lm[4], lm[8]) / span > 0.5;
  return mean > 1.3 && mean < 2.1 && spread < 0.6 && thumbOpen;
}
