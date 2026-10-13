/* RTPI — real-time person identification, fully on-device.
   Pipeline: SCRFD-500M face detection (main thread, every frame) -> 5-point similarity alignment (ArcFace 112x112)
   -> 512-d embedding in a background worker (Accurate: ResNet-50 8-bit, Fast: MobileFaceNet)
   -> cosine match against the local people list. */
'use strict';

const APP_V = '6';
// Never sit silently on the loading screen: show any startup error.
let started = false; // once the camera loop runs, errors are logged instead of taking over the screen
function fatal(msg) {
  const n = document.getElementById('notice'); if (!n || started || /version mismatch/.test(msg)) return;
  n.hidden = false;
  document.getElementById('notice-title').textContent = "RTPI couldn't start";
  document.getElementById('notice-text').textContent = msg;
  const p = document.getElementById('notice-progress'); if (p) p.hidden = true;
  const a = document.getElementById('notice-actions');
  if (a) { a.hidden = false; a.innerHTML = '<button class="btn primary" onclick="location.reload()">Reload</button>'; }
}
addEventListener('error', (e) => fatal(`${e.message || 'Script error'}${e.filename ? ` (${e.filename.split('/').pop()}:${e.lineno})` : ''}`));
addEventListener('unhandledrejection', (e) => fatal(String(e.reason && e.reason.message || e.reason)));

// index.html and app.js must come from the same upload. If the browser mixed an old saved copy
// with a new file, clear the saved app files once and reload.
if (document.documentElement.dataset.v !== APP_V) {
  let tried = false; try { tried = sessionStorage.getItem('rtpi-fix') === APP_V; sessionStorage.setItem('rtpi-fix', APP_V); } catch { }
  if (!tried) {
    Promise.resolve()
      .then(() => navigator.serviceWorker && navigator.serviceWorker.getRegistrations().then((rs) => Promise.all(rs.map((r) => r.unregister()))))
      .then(() => self.caches && caches.keys().then((ks) => Promise.all(ks.filter((k) => !k.startsWith('rtpi-assets') && k !== 'rtpi-share').map((k) => caches.delete(k)))))
      .catch(() => {}).finally(() => location.reload());
  } else {
    fatal(`The app files on your site are from different versions (index.html is v${document.documentElement.dataset.v || '1'}, app.js is v${APP_V}). Upload all the app files (index.html, app.js, rec-worker.js, sw.js, manifest.webmanifest) to GitHub again, wait two minutes, then reload.`);
  }
  throw new Error('version mismatch');
}

const $ = (s) => document.querySelector(s);
const ARC = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
const DIM = 512;
const DEFAULTS = { model: 'accurate', thr_accurate: 0.40, thr_fast: 0.42, margin: 0.05, det: 384, max: 3, scores: false, facing: 'user' };
const MODELS = {
  // split into ~11 MB parts to stay well under GitHub's web-upload size limit
  accurate: { key: 'emb_r50', name: 'Accurate', parts: [1, 2, 3, 4].map((i) => ({ url: `models/r50_int8.onnx.00${i}`, size: i < 4 ? 10947145 : 10947144 })) },
  fast: { key: 'emb', name: 'Fast', parts: [{ url: 'models/w600k_mbf.onnx', size: 13616099 }] },
};
const thr = () => +app.settings['thr_' + app.settings.model];
const embKey = () => MODELS[app.settings.model].key;

const app = {
  det: null, rec: null, wasm: null,
  people: [], mPeople: [], E: new Float32Array(0),
  settings: load('rtpi-settings', DEFAULTS),
  stream: null, running: false,
  tracks: [], nextTrackId: 1, hits: [],
  fps: 0, frameW: 0, frameH: 0,
  last: null,              // last processed frame canvas snapshot info for "add from camera"
  addDraft: null,
};

if (!MODELS[app.settings.model]) app.settings.model = 'accurate';

/* ---------- small utils ---------- */
function load(key, def) { try { return { ...def, ...JSON.parse(localStorage.getItem(key) || '{}') }; } catch { return { ...def }; } }
function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch { /* storage unavailable */ } }
function toast(msg, ms = 2200) { const t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), ms); }
function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function b64ToF32(b64) { const bin = atob(b64); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); }
function f32ToB64(f) { const u8 = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function normalize(v) { let n = 0; for (let i = 0; i < v.length; i++) n += v[i] * v[i]; n = Math.sqrt(n) || 1; for (let i = 0; i < v.length; i++) v[i] /= n; return v; }
function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy; return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

/* ---------- IndexedDB (people stay on this phone) ---------- */
const DB = {
  db: null,
  open() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('rtpi', 1);
      r.onupgradeneeded = () => r.result.createObjectStore('people', { keyPath: 'id' });
      r.onsuccess = () => { this.db = r.result; res(); }; r.onerror = () => rej(r.error);
    });
  },
  tx(mode, fn) {
    return new Promise((res, rej) => {
      const t = this.db.transaction('people', mode); const s = t.objectStore('people'); const out = fn(s);
      t.oncomplete = () => res(out && out.result); t.onerror = () => rej(t.error);
    });
  },
  all() { return this.tx('readonly', (s) => s.getAll()); },
  putMany(arr) { return this.tx('readwrite', (s) => arr.forEach((p) => s.put(p))); },
  del(id) { return this.tx('readwrite', (s) => s.delete(id)); },
  clear() { return this.tx('readwrite', (s) => s.clear()); },
};

// matching index for the active model; people without a fingerprint for it are filled in by backfill()
const okEmb = (v) => v && v.length === DIM;
function rebuildIndex() {
  const key = embKey(), rows = [];
  for (const p of app.people) {
    if (okEmb(p[key])) rows.push([p, p[key]]);
    for (const x of p.extra || []) if (okEmb(x[key])) rows.push([p, x[key]]);
  }
  app.rowP = rows.map((r) => r[0]);
  app.mPeople = [...new Set(app.rowP)];
  const E = new Float32Array(rows.length * DIM);
  rows.forEach(([, v], i) => E.set(v instanceof Float32Array ? v : new Float32Array(v), i * DIM));
  app.E = E;
  app.tracks.forEach((t) => { t.emb = null; t.result = null; t.embAt = 0; });
}

async function reloadPeople() {
  let rows = [];
  try { rows = (await DB.all()) || []; } catch (e) { console.warn('storage unavailable', e); rows = app.people; }
  rows.sort((a, b) => a.name.localeCompare(b.name));
  app.people = rows;
  rebuildIndex();
  $('#count-badge').textContent = rows.length; $('#count-badge').hidden = rows.length === 0;
  renderPeopleList();
  updateNotice();
}

/* ---------- models ---------- */
const ASSET_CACHE = 'rtpi-assets-1';
const BASE_ASSETS = [{ url: 'vendor/ort-wasm-simd-threaded.wasm', size: 11246032 }, { url: 'models/det_500m.onnx', size: 2524817 }];

async function fetchWithProgress(url, onBytes) {
  const ctl = new AbortController(); let timer;
  const arm = () => { clearTimeout(timer); timer = setTimeout(() => ctl.abort(), 30000); };
  arm();
  try {
    const res = await fetch(url, { signal: ctl.signal, cache: 'no-cache' });
    if (!res.ok) throw new Error(res.status === 404 ? `${url} is missing on your site. Upload the ${url.split('/')[0]} folder to GitHub.` : `${url}: HTTP ${res.status}`);
    const reader = res.body.getReader(); const chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; arm(); chunks.push(value); got += value.length; onBytes(value.length); }
    const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('The download stalled. Check your internet connection and reload.');
    throw e;
  } finally { clearTimeout(timer); }
}

async function getAsset(a, onBytes) {
  let cache = null;
  try {
    cache = await caches.open(ASSET_CACHE);
    const hit = await cache.match(a.url);
    if (hit) { const b = await hit.arrayBuffer(); if (b.byteLength === a.size) { onBytes(b.byteLength); return b; } }
  } catch { cache = null; }
  const buf = await fetchWithProgress(a.url, onBytes);
  if (buf.byteLength !== a.size) throw new Error(`${a.url} on your site is ${(buf.byteLength / 1e6).toFixed(2)} MB but should be ${(a.size / 1e6).toFixed(2)} MB. Upload that file to GitHub again.`);
  try { await cache?.put(a.url, new Response(buf.slice(0), { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* storage full or blocked: works, just re-downloads next time */ }
  return buf;
}

function joinBuffers(bufs) {
  if (bufs.length === 1) return bufs[0];
  const out = new Uint8Array(bufs.reduce((t, b) => t + b.byteLength, 0)); let o = 0;
  for (const b of bufs) { out.set(new Uint8Array(b), o); o += b.byteLength; } return out.buffer;
}

// Recogniser: one background worker per model. embed() queues requests; `pending` lets the live loop avoid piling up.
function makeRecognizer(modelBuf) {
  return new Promise((resolve, reject) => {
    const w = new Worker('rec-worker.js?v=' + APP_V); const waiting = new Map(); let seq = 0;
    const rec = {
      pending: 0,
      embed(x, n) {
        return new Promise((res, rej) => { const id = ++seq; waiting.set(id, { res, rej, n }); rec.pending++; w.postMessage({ type: 'embed', id, x, n }, [x.buffer]); });
      },
      close() { w.terminate(); },
    };
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') return resolve(rec);
      const job = waiting.get(m.id);
      if (!job) { if (m.type === 'error') reject(new Error(m.message)); return; }
      waiting.delete(m.id); rec.pending--;
      if (m.type === 'error') return job.rej(new Error(m.message));
      job.res(Array.from({ length: job.n }, (_, k) => normalize(m.data.slice(k * DIM, (k + 1) * DIM))));
    };
    w.onerror = (e) => reject(new Error('Face recogniser failed to start: ' + (e.message || 'worker error')));
    w.postMessage({ type: 'init', base: new URL('./', location.href).href, wasm: app.wasm.slice(0), model: modelBuf }, [modelBuf]);
  });
}

async function loadRecognizer(model, tick) {
  const bufs = await Promise.all(MODELS[model].parts.map((a) => getAsset(a, tick)));
  const rec = await makeRecognizer(joinBuffers(bufs));
  if (app.rec) app.rec.close();
  app.rec = rec; app.recModel = model;
}

function progressTicker(total) {
  let got = 0; const bar = $('#notice-progress > div'), txt = $('#notice-text');
  return (n) => {
    if (app.loadError) return;
    got += n; bar.style.width = Math.min(100, (got / total) * 100).toFixed(1) + '%';
    txt.textContent = `${(got / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(0)} MB. Downloaded once, then kept on this phone.`;
  };
}

async function loadModels() {
  ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
  ort.env.logLevel = 'error';
  ort.env.wasm.numThreads = 1;
  const model = app.settings.model;
  const total = [...BASE_ASSETS, ...MODELS[model].parts].reduce((t, a) => t + a.size, 0);
  const tick = progressTicker(total);
  const [wasmBuf, detBuf] = await Promise.all(BASE_ASSETS.map((a) => getAsset(a, tick)));
  app.wasm = wasmBuf;
  ort.env.wasm.wasmBinary = wasmBuf.slice(0);
  const recP = loadRecognizer(model, tick);
  app.det = await ort.InferenceSession.create(detBuf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 });
  $('#notice-title').textContent = 'Preparing face models';
  await recP;
}

// SCRFD-10G: more precise on small, blurry or turned faces. Used for photos, not live video.
async function getDet10() {
  if (app.det10) return app.det10;
  if (!app.det10P) app.det10P = (async () => {
    const bufs = await Promise.all([{ url: 'models/det_10g.onnx.001', size: 8461914 }, { url: 'models/det_10g.onnx.002', size: 8461913 }].map((a) => getAsset(a, () => {})));
    const buf = joinBuffers(bufs);
    app.det10 = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 });
    return app.det10;
  })().catch((e) => { app.det10P = null; throw e; });
  return app.det10P;
}
// detect on a still photo with the best detector available
async function detectPhoto(c, S = 640, thresh = 0.5) {
  let sess = app.det;
  try { sess = await getDet10(); } catch (e) { console.warn('sharper detector unavailable, using the fast one', e); }
  return detect(c, c.width, c.height, S, thresh, sess);
}
async function loadImageCanvas(blob, maxSide = 2000) {
  const bmp = await createImageBitmap(blob);
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height)), W = Math.round(bmp.width * k), H = Math.round(bmp.height * k);
  const c = canvas(W, H); c.getContext('2d').drawImage(bmp, 0, 0, W, H); bmp.close?.();
  return c;
}
const largest = (faces) => faces.slice().sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]))[0];

async function switchModel(model) {
  if (model === app.recModel) return;
  const total = MODELS[model].parts.reduce((t, a) => t + a.size, 0); let got = 0;
  $('#s-model').disabled = true;
  try {
    await loadRecognizer(model, (n) => { got += n; $('#model-note').textContent = `Loading ${MODELS[model].name} model… ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`; });
    app.settings.model = model; save('rtpi-settings', app.settings);
    rebuildIndex(); syncSettings(); toast(`${MODELS[model].name} model on`);
    backfill();
  } catch (e) { toast(e.message || 'Could not load that model', 4000); $('#s-model').value = app.recModel; }
  $('#s-model').disabled = false; syncSettings();
}

/* ---------- detection (SCRFD) ---------- */
const detCanvas = canvas(640, 640), detCtx = detCanvas.getContext('2d', { willReadFrequently: true });

async function detect(src, W, H, S, thresh = 0.5, sess = app.det) {
  if (detCanvas.width !== S) { detCanvas.width = detCanvas.height = S; }
  const s = S / Math.max(W, H);
  detCtx.fillStyle = '#000'; detCtx.fillRect(0, 0, S, S);
  detCtx.drawImage(src, 0, 0, W, H, 0, 0, Math.round(W * s), Math.round(H * s));
  const px = detCtx.getImageData(0, 0, S, S).data; const n = S * S; const x = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { x[i] = (px[j] - 127.5) / 128; x[n + i] = (px[j + 1] - 127.5) / 128; x[2 * n + i] = (px[j + 2] - 127.5) / 128; }
  const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, S, S]) });
  const o = sess.outputNames.map((k) => out[k].data);
  const faces = [];
  [8, 16, 32].forEach((stride, k) => {
    const sc = o[k], bb = o[k + 3], kp = o[k + 6], fs = S / stride;
    for (let idx = 0; idx < sc.length; idx++) {
      if (sc[idx] < thresh) continue;
      const cell = idx >> 1, cx = (cell % fs) * stride, cy = Math.floor(cell / fs) * stride;
      const box = [(cx - bb[idx * 4] * stride) / s, (cy - bb[idx * 4 + 1] * stride) / s, (cx + bb[idx * 4 + 2] * stride) / s, (cy + bb[idx * 4 + 3] * stride) / s];
      const pts = []; for (let p = 0; p < 5; p++) pts.push([(cx + kp[idx * 10 + p * 2] * stride) / s, (cy + kp[idx * 10 + p * 2 + 1] * stride) / s]);
      faces.push({ score: sc[idx], box, pts });
    }
  });
  faces.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const f of faces) if (keep.every((g) => iou(f.box, g.box) < 0.4)) keep.push(f);
  return keep;
}

/* ---------- alignment + embedding ---------- */
function similarity(src) { // least-squares similarity transform src -> ARC
  let mx = 0, my = 0, ux = 0, uy = 0; for (let i = 0; i < 5; i++) { mx += src[i][0]; my += src[i][1]; ux += ARC[i][0]; uy += ARC[i][1]; }
  mx /= 5; my /= 5; ux /= 5; uy /= 5;
  let den = 0, na = 0, nb = 0;
  for (let i = 0; i < 5; i++) {
    const px = src[i][0] - mx, py = src[i][1] - my, qx = ARC[i][0] - ux, qy = ARC[i][1] - uy;
    den += px * px + py * py; na += px * qx + py * qy; nb += px * qy - py * qx;
  }
  const a = na / den, b = nb / den;
  return { a, b, tx: ux - (a * mx - b * my), ty: uy - (b * mx + a * my) };
}
const alignCanvas = canvas(112, 112), alignCtx = alignCanvas.getContext('2d', { willReadFrequently: true });

function alignedPixels(src, pts) {
  const m = similarity(pts);
  alignCtx.setTransform(1, 0, 0, 1, 0, 0); alignCtx.fillStyle = '#808080'; alignCtx.fillRect(0, 0, 112, 112);
  alignCtx.imageSmoothingEnabled = true; alignCtx.imageSmoothingQuality = 'high';
  alignCtx.setTransform(m.a, m.b, -m.b, m.a, m.tx, m.ty);
  alignCtx.drawImage(src, 0, 0);
  alignCtx.setTransform(1, 0, 0, 1, 0, 0);
  return alignCtx.getImageData(0, 0, 112, 112).data;
}

function alignedBatch(src, faces) {
  const n = 112 * 112, x = new Float32Array(faces.length * 3 * n);
  faces.forEach((f, k) => {
    const px = alignedPixels(src, f.pts), o = k * 3 * n;
    for (let i = 0, j = 0; i < n; i++, j += 4) { x[o + i] = (px[j] - 127.5) / 127.5; x[o + n + i] = (px[j + 1] - 127.5) / 127.5; x[o + 2 * n + i] = (px[j + 2] - 127.5) / 127.5; }
  });
  return x;
}
async function embedMany(src, faces) {
  if (!faces.length) return [];
  return app.rec.embed(alignedBatch(src, faces), faces.length);
}

// best score per person over all their photos, then the k best people
function matchTop(e, k = 3) {
  if (!okEmb(e)) return [];
  const best = new Map(), N = app.rowP.length;
  for (let r = 0; r < N; r++) {
    let s = 0; const o = r * DIM; for (let i = 0; i < DIM; i++) s += app.E[o + i] * e[i];
    const p = app.rowP[r], cur = best.get(p);
    if (cur === undefined || s > cur) best.set(p, s);
  }
  return [...best].map(([p, s]) => ({ p, s })).sort((a, b) => b.s - a.s).slice(0, k);
}

function match(e) {
  const [a, b] = matchTop(e, 2);
  const ok = !!a && a.s >= thr() && a.s - Math.max(b ? b.s : 0, 0) >= app.settings.margin;
  return { person: ok ? a.p : null, closest: a ? a.p : null, score: a ? a.s : 0, second: b ? b.s : 0 };
}

/* ---------- camera ---------- */
const video = $('#video');
async function startCamera() {
  stopCamera();
  const facing = app.settings.facing;
  try {
    app.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } } });
  } catch (e) {
    showNotice('Camera not available', e.name === 'NotAllowedError'
      ? 'Allow camera access for this site in your browser settings, then reload.'
      : `Your browser couldn't open the camera (${e.name}). It needs Chrome on Android over https.`, [{ label: 'Try again', primary: true, fn: startCamera }]);
    return false;
  }
  video.srcObject = app.stream;
  video.classList.toggle('mirror', facing === 'user');
  await video.play().catch(() => {});
  app.tracks = [];
  try { app.wake = await navigator.wakeLock?.request('screen'); } catch { /* optional */ }
  return true;
}
function stopCamera() { if (app.stream) app.stream.getTracks().forEach((t) => t.stop()); app.stream = null; }

/* ---------- main loop ---------- */
const frameCanvas = canvas(16, 16), frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });
const MAX_SIDE = 960;

async function loop() {
  if (app.running) return; app.running = true;
  let lastT = performance.now();
  while (app.running) {
    if (app.paused || !app.stream || video.readyState < 2 || !video.videoWidth || document.hidden) { await sleep(120); continue; }
    const vw = video.videoWidth, vh = video.videoHeight, k = Math.min(1, MAX_SIDE / Math.max(vw, vh));
    const W = Math.round(vw * k), H = Math.round(vh * k);
    if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
    frameCtx.drawImage(video, 0, 0, W, H);
    app.frameW = W; app.frameH = H; app.frameScale = k;
    try {
      const faces = (await detect(frameCanvas, W, H, +app.settings.det)).filter((f) => f.box[2] - f.box[0] > 24);
      faces.sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]));
      associate(faces.slice(0, 8));
      scheduleRecognition();
    } catch (e) { console.error(e); }
    draw();
    const now = performance.now(); app.fps = app.fps * 0.8 + (1000 / Math.max(1, now - lastT)) * 0.2; lastT = now;
    setStatus();
    await new Promise((r) => requestAnimationFrame(r));
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// follow faces from frame to frame by overlap
function associate(faces) {
  const used = new Set(); const next = [];
  faces.forEach((f) => {
    let bestT = null, bestI = 0.25;
    for (const t of app.tracks) { if (used.has(t)) continue; const v = iou(t.box, f.box); if (v > bestI) { bestI = v; bestT = t; } }
    const t = bestT || { id: app.nextTrackId++, emb: null, seen: 0, embAt: 0, busy: false };
    used.add(t); t.box = f.box; t.pts = f.pts; t.missed = 0;
    next.push(t);
  });
  // keep briefly-lost tracks for a couple of frames so labels don't flicker
  for (const t of app.tracks) if (!used.has(t) && ++t.missed <= 2) next.push(t);
  app.tracks = next;
}

// smooth the identity over several recognitions: running mean of embeddings, re-normalised
function applyEmbedding(t, e) {
  let same = 0; if (t.emb) for (let d = 0; d < DIM; d++) same += t.emb[d] * e[d];
  if (!t.emb || same < 0.4) { t.emb = Float32Array.from(e); t.seen = 0; } // a different face in this spot: start fresh
  else { const w = Math.min(t.seen, 4); for (let d = 0; d < DIM; d++) t.emb[d] = t.emb[d] * w + e[d]; normalize(t.emb); }
  t.seen++; t.raw = e; t.result = match(t.emb);
}

// Accurate: one face at a time, newest/unnamed first, each refreshed every ~1.5 s.
// Fast: every face (up to the setting) on every frame.
function scheduleRecognition() {
  if (!app.rec || app.rec.pending > 0) return;
  const live = app.tracks.filter((t) => !t.missed && !t.busy);
  if (!live.length) return;
  const now = performance.now(); let pick;
  if (app.recModel === 'fast') pick = live.slice(0, +app.settings.max);
  else {
    const due = live.filter((t) => !t.emb || now - t.embAt > 1500);
    due.sort((a, b) => (a.emb ? 1 : 0) - (b.emb ? 1 : 0) || a.embAt - b.embAt);
    pick = due.slice(0, 1);
  }
  if (!pick.length) return;
  const x = alignedBatch(frameCanvas, pick);
  pick.forEach((t) => { t.busy = true; });
  const model = app.recModel;
  app.rec.embed(x, pick.length).then((embs) => {
    pick.forEach((t, i) => { t.busy = false; t.embAt = performance.now(); if (model === app.recModel) applyEmbedding(t, embs[i]); });
    draw();
  }).catch((e) => { pick.forEach((t) => (t.busy = false)); console.error(e); });
}

/* ---------- overlay ---------- */
const overlay = $('#overlay'), octx = overlay.getContext('2d');
function sizeOverlay() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlay.width = Math.round(innerWidth * dpr); overlay.height = Math.round(innerHeight * dpr);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener('resize', sizeOverlay);

function toScreen(box) {
  const cw = innerWidth, ch = innerHeight, vw = video.videoWidth, vh = video.videoHeight, k = app.frameScale || 1;
  const sc = Math.max(cw / vw, ch / vh), ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
  let x1 = (box[0] / k) * sc + ox, x2 = (box[2] / k) * sc + ox;
  const y1 = (box[1] / k) * sc + oy, y2 = (box[3] / k) * sc + oy;
  if (app.settings.facing === 'user') { const a = cw - x2, b = cw - x1; x1 = a; x2 = b; }
  return [x1, y1, x2, y2];
}

function fit(text, maxW) {
  if (octx.measureText(text).width <= maxW) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (octx.measureText(text.slice(0, mid) + '…').width <= maxW) lo = mid; else hi = mid - 1; }
  return text.slice(0, lo).trimEnd() + '…';
}

function draw() {
  const cw = innerWidth, ch = innerHeight;
  octx.clearRect(0, 0, cw, ch); app.hits = [];
  const css = getComputedStyle(document.documentElement);
  const known = css.getPropertyValue('--known').trim(), unknown = css.getPropertyValue('--unknown').trim();
  for (const t of app.tracks) {
    if (t.missed > 0 && !t.result?.person) continue;
    const [x1, y1, x2, y2] = toScreen(t.box); const w = x2 - x1, h = y2 - y1;
    const r = t.result || {}; const isKnown = !!r.person; const col = isKnown ? known : unknown;
    // corner brackets
    const L = Math.max(12, Math.min(w, h) * 0.22);
    octx.strokeStyle = col; octx.lineWidth = isKnown ? 3 : 2; octx.lineCap = 'round'; octx.beginPath();
    [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]].forEach(([x, y, dx, dy]) => { octx.moveTo(x + dx * L, y); octx.lineTo(x, y); octx.lineTo(x, y + dy * L); });
    octx.stroke();
    // name tag
    const tagW = Math.max(150, Math.min(280, w * 1.4)), pad = 10;
    let title, line2;
    if (isKnown) { title = r.person.name; line2 = r.person.headline || r.person.organisation || ''; }
    else if (!app.people.length) { title = 'Face found'; line2 = 'Import your face pack to name people'; }
    else if (!t.result) { title = 'Identifying…'; line2 = ''; }
    else { title = 'Unknown'; line2 = app.settings.scores && r.closest ? `closest: ${r.closest.name}` : 'Tap to add'; }
    if (app.settings.scores && r.closest && t.result) title += `  ${r.score.toFixed(2)}`;
    octx.font = '700 15px ' + css.getPropertyValue('--ui'); const t1 = fit(title, tagW - pad * 2);
    octx.font = '13px ' + css.getPropertyValue('--ui'); const t2 = line2 ? fit(line2, tagW - pad * 2) : '';
    const tagH = t2 ? 46 : 28;
    let tx = (x1 + x2) / 2 - tagW / 2; tx = Math.max(8, Math.min(cw - tagW - 8, tx));
    let ty = y2 + 10; if (ty + tagH > ch - 110) ty = Math.max(70, y1 - tagH - 10);
    octx.fillStyle = isKnown ? known : 'rgba(23,27,34,.86)';
    octx.beginPath(); octx.roundRect(tx, ty, tagW, tagH, 10); octx.fill();
    octx.fillStyle = isKnown ? css.getPropertyValue('--known-ink') : css.getPropertyValue('--text');
    octx.font = '700 15px ' + css.getPropertyValue('--ui'); octx.textBaseline = 'top'; octx.fillText(t1, tx + pad, ty + 7);
    if (t2) { octx.globalAlpha = 0.8; octx.font = '13px ' + css.getPropertyValue('--ui'); octx.fillText(t2, tx + pad, ty + 26); octx.globalAlpha = 1; }
    app.hits.push({ rect: [Math.min(x1, tx), Math.min(y1, ty), Math.max(x2, tx + tagW), Math.max(y2, ty + tagH)], track: t });
  }
}

overlay.addEventListener('click', (ev) => {
  const h = app.hits.find(({ rect: [a, b, c, d] }) => ev.clientX >= a && ev.clientX <= c && ev.clientY >= b && ev.clientY <= d);
  if (!h) return;
  const r = h.track.result || {};
  if (r.person) openPerson(r.person, r.score, false, snapshotTrack(h.track));
  else openAdd(snapshotTrack(h.track));
});

/* ---------- status + notice ---------- */
function setStatus() {
  const n = app.people.length, named = app.tracks.filter((t) => t.result?.person && !t.missed).length;
  const bf = app.backfill ? ` · updating ${app.backfill.done}/${app.backfill.total}` : '';
  $('#status').innerHTML = `<b>${n}</b> people · ${MODELS[app.recModel || app.settings.model].name} · ${app.fps.toFixed(0)} fps${named ? ` · <b>${named}</b> named` : ''}${bf}`;
}
function showNotice(title, text, actions = [], progress = false) {
  $('#notice').hidden = false; $('#notice-title').textContent = title; $('#notice-text').textContent = text;
  $('#notice-progress').hidden = !progress;
  const box = $('#notice-actions'); box.innerHTML = ''; box.hidden = !actions.length;
  actions.forEach((a) => { const b = document.createElement('button'); b.className = 'btn' + (a.primary ? ' primary' : ''); b.textContent = a.label; b.onclick = a.fn; box.append(b); });
}
function hideNotice() { $('#notice').hidden = true; }
function updateNotice() {
  if (!app.det) return; // still loading
  if (!app.stream) return;
  if (!app.people.length) showNotice('No people loaded yet', 'Import the face_pack.json file from your computer. It stays on this phone.', [{ label: 'Import face pack', primary: true, fn: () => $('#file-pack').click() }, { label: 'Not now', fn: hideNotice }]);
  else hideNotice();
}

/* ---------- sheets ---------- */
let openSheetEl = null;
function openSheet(id) { closeSheet(); openSheetEl = $(id); openSheetEl.hidden = false; $('#scrim').hidden = false; app.paused = id === '#sheet-photo'; if (app.paused) { app.tracks = []; draw(); } }
function closeSheet() { if (openSheetEl) openSheetEl.hidden = true; openSheetEl = null; $('#scrim').hidden = true; if (app.paused) app.tracks = []; app.paused = false; }
$('#scrim').onclick = closeSheet;
addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });

function openPerson(p, score, fromPhoto = false, face = null) {
  const t0 = thr();
  app.cardPerson = p; app.cardFace = face && okEmb(face.emb) ? face : null;
  const n = 1 + (p.extra || []).length;
  $('#person-body').innerHTML = `${fromPhoto ? '<button class="btn back" id="btn-back-photo">← Back to photo</button>' : ''}
    <div class="person">
      <img src="${esc(p.thumb)}" alt="">
      <div class="who">
        <div class="name">${esc(p.name)}${p.source === 'added' ? '<span class="tag">added</span>' : ''}</div>
        ${p.headline ? `<p class="head">${esc(p.headline)}</p>` : ''}
        <div class="meta">${[p.organisation, p.location].filter(Boolean).map(esc).join(' · ')}</div>
      </div>
    </div>
    ${score != null ? `<div class="match"><span>match</span><div class="meter"><div style="width:${Math.max(0, Math.min(1, score)) * 100}%"></div><i style="left:${t0 * 100}%"></i></div><span>${score.toFixed(2)}</span></div>` : ''}
    ${p.link ? `<a class="link" href="${esc(p.link)}" target="_blank" rel="noopener">${esc(p.link.replace(/^https?:\/\/(www\.)?/, ''))}</a>` : ''}
    <div class="photos">
      <div class="photos-head"><span>Photos used for recognition</span><span class="count">${n}</span></div>
      <div class="strip" id="photo-strip"></div>
      <div class="row">
        ${app.cardFace ? '<button class="btn primary" id="btn-save-face">Save this face to their photos</button>' : ''}
        <button class="btn" id="btn-add-extra">Add a photo</button>
      </div>
      <small class="hint">More photos of the same person, from different angles and lighting, make recognition more reliable.</small>
    </div>
    <button class="btn danger" id="btn-del-person">Remove from this phone</button>`;
  renderStrip(p);
  if (app.cardFace) $('#btn-save-face').onclick = () => addExtra(p, app.cardFace);
  $('#btn-add-extra').onclick = () => $('#file-extra').click();
  if (fromPhoto) $('#btn-back-photo').onclick = () => openSheet('#sheet-photo');
  armButton($('#btn-del-person'), 'Tap again to remove', async () => { await DB.del(p.id); await reloadPeople(); closeSheet(); toast(`Removed ${p.name}`); });
  openSheet('#sheet-person');
}

function renderStrip(p) {
  const strip = $('#photo-strip'); if (!strip) return; strip.innerHTML = '';
  const main = document.createElement('figure'); main.innerHTML = `<img src="${esc(p.thumb)}" alt=""><figcaption>Main</figcaption>`; strip.append(main);
  (p.extra || []).forEach((x) => {
    const f = document.createElement('figure');
    f.innerHTML = `<img src="${esc(x.thumb)}" alt=""><button class="rm" aria-label="Remove this photo">×</button>`;
    const b = f.querySelector('.rm');
    armButton(b, '?', async () => {
      p.extra = p.extra.filter((y) => y !== x); await DB.putMany([p]).catch(() => {});
      rebuildIndex(); renderStrip(p); toast('Photo removed');
    });
    strip.append(f);
  });
}

// add another photo (template) to a person
async function addExtra(p, face) {
  const key = face.key || embKey();
  const before = matchTop(face.emb, 50).find((t) => t.p === p);
  const x = { id: 'ph:' + Date.now().toString(36), thumb: face.thumb, added: new Date().toISOString().slice(0, 10) };
  x[key] = face.emb;
  p.extra = [...(p.extra || []), x];
  await DB.putMany([p]).catch(() => {});
  rebuildIndex(); renderStrip(p);
  if (app.cardFace === face) { app.cardFace = null; $('#btn-save-face')?.remove(); }
  const cnt = $('.photos .count'); if (cnt) cnt.textContent = 1 + p.extra.length;
  if (before && before.s < 0.2) toast(`Saved, but this face looks quite different from ${p.name}'s other photos (${before.s.toFixed(2)}). If it's someone else, remove it.`, 5000);
  else toast(`Saved. ${p.name} now has ${1 + p.extra.length} photos.`);
}

function armButton(btn, armedLabel, fn) {
  const label = btn.textContent; let armed = false, t;
  btn.onclick = () => {
    if (armed) { clearTimeout(t); fn(); return; }
    armed = true; btn.classList.add('armed'); btn.textContent = armedLabel;
    t = setTimeout(() => { armed = false; btn.classList.remove('armed'); btn.textContent = label; }, 3500);
  };
}

/* people list */
function renderPeopleList() {
  const q = ($('#people-search').value || '').toLowerCase().trim();
  const list = $('#people-list'); list.innerHTML = '';
  const rows = app.people.filter((p) => !q || [p.name, p.headline, p.organisation, p.location].join(' ').toLowerCase().includes(q));
  $('#people-title').textContent = `People (${app.people.length})`;
  const frag = document.createDocumentFragment();
  rows.slice(0, 400).forEach((p) => {
    const b = document.createElement('button');
    b.innerHTML = `<img loading="lazy" src="${esc(p.thumb)}" alt=""><span class="t"><div class="n">${esc(p.name)}</div><div class="h">${esc(p.headline || p.organisation || '')}</div></span>`;
    b.onclick = () => openPerson(p);
    frag.append(b);
  });
  list.append(frag);
  if (!app.people.length) list.innerHTML = '<p class="sub">Nobody yet. Import face_pack.json, or add people one at a time with the + button.</p>';
}
$('#people-search').addEventListener('input', renderPeopleList);
$('#btn-people').onclick = () => { renderPeopleList(); openSheet('#sheet-people'); };
$('#btn-import').onclick = () => $('#file-pack').click();
$('#file-pack').onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const pack = JSON.parse(await f.text());
    if (pack.format !== 'face-pack' || !Array.isArray(pack.people)) throw new Error('This file is not an RTPI face pack.');
    const have = new Map(app.people.map((p) => [p.id, p]));
    const rows = pack.people.map((p) => {
      const row = { ...(have.get(String(p.id)) || {}), id: String(p.id), name: p.name, headline: p.headline || '', organisation: p.organisation || '', location: p.location || '', link: p.link || '', thumb: p.thumb || '', source: p.source || 'pack' };
      for (const k of ['emb', 'emb_r50']) if (p[k]) { const v = b64ToF32(p[k]); if (v.length !== DIM) throw new Error(`Bad face data for ${p.name}.`); row[k] = v; }
      if (Array.isArray(p.extra)) {
        // photos from the computer replace earlier ones from the computer; photos added on this phone are kept
        const mine = (row.extra || []).filter((x) => !String(x.id).startsWith('pc:'));
        const theirs = p.extra.map((x) => { const y = { id: String(x.id || 'pc:' + Math.random().toString(36).slice(2)), thumb: x.thumb || '', added: x.added || '' };
          for (const k of ['emb', 'emb_r50']) if (x[k]) { const v = b64ToF32(x[k]); if (v.length === DIM) y[k] = v; } return y; });
        const ids = new Set(theirs.map((x) => x.id));
        row.extra = [...mine.filter((x) => !ids.has(x.id)), ...theirs];
      }
      return row;
    });
    await DB.putMany(rows).catch(() => { app.people = rows; });
    await reloadPeople();
    toast(`Imported ${rows.length} people`);
    backfill();
  } catch (err) { toast(err.message || 'Could not read that file', 4000); }
};
$('#btn-export').onclick = () => {
  const enc = (v) => v ? f32ToB64(v instanceof Float32Array ? v : new Float32Array(v)) : undefined;
  const pack = { format: 'face-pack', version: 2, models: ['w600k_mbf', 'w600k_r50'], dim: DIM, created: new Date().toISOString().slice(0, 16).replace('T', ' '),
    people: app.people.map((p) => ({ id: p.id, name: p.name, headline: p.headline, organisation: p.organisation, location: p.location, link: p.link, thumb: p.thumb, source: p.source, emb: enc(p.emb), emb_r50: enc(p.emb_r50),
      extra: (p.extra || []).map((x) => ({ id: x.id, thumb: x.thumb, added: x.added, emb: enc(x.emb), emb_r50: enc(x.emb_r50) })) })) };
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(pack)], { type: 'application/json' }));
  a.download = `face_pack_${pack.people.length}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};
armButton($('#btn-clear'), 'Tap again to delete everyone', async () => { await DB.clear(); await reloadPeople(); toast('Deleted everyone from this phone'); });

/* add person */
function snapshotTrack(t) {
  if (!t || !t.box) return null;
  const [x1, y1, x2, y2] = t.box, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, half = Math.max(x2 - x1, y2 - y1) * 0.9;
  const c = canvas(160, 160), g = c.getContext('2d');
  g.drawImage(frameCanvas, cx - half, cy - half, half * 2, half * 2, 0, 0, 160, 160);
  if (!t.raw) return null;
  return { thumb: c.toDataURL('image/jpeg', 0.82), emb: Float32Array.from(t.raw), key: embKey() };
}
function openAdd(draft) {
  app.addDraft = draft || null;
  $('#add-form').reset(); $('#add-existing-q').value = '';
  $('#add-preview').src = draft ? draft.thumb : '';
  $('#add-sub').textContent = draft ? 'Face captured. Add their details.' : 'Use the face in front of the camera now, or pick a photo.';
  renderAddSuggest();
  openSheet('#sheet-add');
}
// "Already in your list?": closest faces first, or search by name
function renderAddSuggest() {
  const box = $('#add-existing-list'); box.innerHTML = '';
  const q = $('#add-existing-q').value.trim().toLowerCase();
  let rows;
  if (q) rows = app.people.filter((p) => [p.name, p.headline, p.organisation].join(' ').toLowerCase().includes(q)).slice(0, 6).map((p) => ({ p }));
  else rows = app.addDraft ? matchTop(app.addDraft.emb, 3) : [];
  $('#add-existing').hidden = !app.people.length;
  rows.forEach(({ p, s }) => {
    const b = document.createElement('button'); b.type = 'button';
    b.innerHTML = `<img src="${esc(p.thumb)}" alt=""><span class="t"><div class="n">${esc(p.name)}</div><div class="h">${s != null ? `looks ${s.toFixed(2)} similar · ` : ''}${esc(p.headline || p.organisation || '')}</div></span>`;
    b.onclick = async () => {
      if (!app.addDraft) { toast('Capture a face or pick a photo first.'); return; }
      await addExtra(p, app.addDraft); app.addDraft = null; closeSheet();
    };
    box.append(b);
  });
  if (!rows.length) box.innerHTML = `<p class="sub">${q ? 'No one by that name.' : 'Capture a face to see who it looks like.'}</p>`;
}
$('#add-existing-q').addEventListener('input', renderAddSuggest);
$('#add-existing-q').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.preventDefault(); });
$('#btn-add').onclick = () => {
  const t = app.tracks.filter((t) => !t.missed).sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
  openAdd(t && !t.result?.person ? snapshotTrack(t) : null);
};
$('#add-from-camera').onclick = () => {
  const t = app.tracks.filter((t) => !t.missed).sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
  if (!t) { toast('No face in view. Point the camera at the person.'); return; }
  if (!t.raw) { toast('Hold still for a second while the face is read, then try again.'); return; }
  app.addDraft = snapshotTrack(t); $('#add-preview').src = app.addDraft.thumb; $('#add-sub').textContent = 'Face captured. Add their details.'; renderAddSuggest();
};
$('#file-extra').onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ''; const p = app.cardPerson; if (!f || !p) return;
  try {
    toast('Reading the photo…', 8000);
    const c = await loadImageCanvas(f, 1600);
    const faces = await detectPhoto(c, 640);
    if (!faces.length) { toast('No face found in that photo. Try a clearer, front-facing one.', 3500); return; }
    const f0 = largest(faces); const [emb] = await embedMany(c, [f0]);
    await addExtra(p, { thumb: cropData(c, f0.box), emb, key: embKey() });
    if (faces.length > 1) toast(`Several faces in that photo; used the largest. ${p.name} now has ${1 + p.extra.length} photos.`, 4000);
  } catch (err) { console.error(err); toast('Could not read that photo.'); }
};
$('#add-from-photo').onclick = () => $('#file-photo').click();
$('#file-photo').onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    $('#add-sub').textContent = 'Reading the photo…';
    const c = await loadImageCanvas(f, 1600);
    const faces = await detectPhoto(c, 640);
    if (!faces.length) { toast('No face found in that photo. Try a clearer, front-facing one.', 3500); $('#add-sub').textContent = 'No face found. Try another photo.'; return; }
    const f0 = largest(faces);
    const [emb] = await embedMany(c, [f0]);
    app.addDraft = { thumb: cropData(c, f0.box), emb, key: embKey() };
    $('#add-preview').src = app.addDraft.thumb; $('#add-sub').textContent = faces.length > 1 ? 'Several faces found; used the largest.' : 'Face found. Add their details.';
    renderAddSuggest();
  } catch (err) { console.error(err); toast('Could not read that photo.'); }
};
$('#add-form').onsubmit = async (e) => {
  e.preventDefault();
  if (!app.addDraft) { toast('Capture a face or pick a photo first.'); return; }
  const name = $('#add-name').value.trim(); if (!name) return;
  const dup = app.people.length ? match(app.addDraft.emb) : null;
  const p = { id: 'u' + Date.now().toString(36), name, headline: $('#add-headline').value.trim(), organisation: $('#add-org').value.trim(),
    location: $('#add-location').value.trim(), link: $('#add-link').value.trim(), thumb: app.addDraft.thumb, source: 'added' };
  p[app.addDraft.key || embKey()] = app.addDraft.emb;
  await DB.putMany([p]).catch(() => app.people.push(p));
  await reloadPeople(); closeSheet();
  toast(dup?.person ? `Saved ${name}. Looks similar to ${dup.person.name}; check it's not a duplicate.` : `Saved ${name}`, dup?.person ? 4500 : 2200);
};

/* ---------- check a photo ---------- */
function cropData(src, box, size = 160) {
  const [x1, y1, x2, y2] = box, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, half = Math.max(x2 - x1, y2 - y1) * 0.9;
  const c = canvas(size, size); const g = c.getContext('2d'); g.fillStyle = '#20262f'; g.fillRect(0, 0, size, size);
  g.drawImage(src, cx - half, cy - half, half * 2, half * 2, 0, 0, size, size);
  return c.toDataURL('image/jpeg', 0.82);
}

async function checkPhoto(blob) {
  openSheet('#sheet-photo');
  if (!app.det) { $('#photo-sub').textContent = 'The face models are still loading. Try again in a moment.'; return; }
  $('#photo-sub').textContent = 'Looking for faces…'; $('#photo-results').innerHTML = ''; $('#photo-wrap').hidden = true;
  let c;
  try { c = await loadImageCanvas(blob, 2000); } catch { $('#photo-sub').textContent = "That file couldn't be opened as a photo."; return; }
  const W = c.width, H = c.height;
  let faces = [];
  try {
    if (!app.det10) $('#photo-sub').textContent = 'Loading the sharper face finder (17 MB, first time only)…';
    faces = await detectPhoto(c, 640, 0.5);
    $('#photo-sub').textContent = 'Looking for faces…';
    if (faces.length < 3 && Math.max(W, H) > 900) { const more = await detectPhoto(c, 960, 0.45); if (more.length > faces.length) faces = more; } // small faces in big photos
  } catch (e) { console.error(e); }
  faces = faces.slice(0, 40).sort((a, b) => a.box[0] - b.box[0]);
  if (faces.length) $('#photo-sub').textContent = `Recognising ${faces.length} face${faces.length > 1 ? 's' : ''}…`;
  const key = embKey();
  let embs = [];
  try { embs = faces.length ? await embedMany(c, faces) : []; } catch (e) { $('#photo-sub').textContent = 'Recognition failed: ' + e.message; return; }
  // each person can only appear once: strongest match claims them
  const res = faces.map((f, i) => ({ f, e: embs[i], key, top: matchTop(embs[i], 3), person: null, crop: cropData(c, f.box) }));
  const taken = new Set();
  [...res].sort((a, b) => (b.top[0]?.s || 0) - (a.top[0]?.s || 0)).forEach((r) => {
    const [a, b] = r.top;
    if (a && a.s >= thr() && a.s - Math.max(b?.s || 0, 0) >= app.settings.margin && !taken.has(a.p.id)) { r.person = a.p; r.score = a.s; taken.add(a.p.id); }
  });
  app.photo = { c, W, H, res };
  renderPhoto();
}

function renderPhoto() {
  const { c, W, H, res } = app.photo;
  const known = getComputedStyle(document.documentElement).getPropertyValue('--known').trim();
  const grey = getComputedStyle(document.documentElement).getPropertyValue('--unknown').trim();
  const named = res.filter((r) => r.person).length;
  $('#photo-sub').textContent = !res.length ? 'No faces found. Try a clearer photo where faces are larger.'
    : !app.people.length ? `${res.length} face${res.length > 1 ? 's' : ''} found. Import your face pack to name them.`
    : `${res.length} face${res.length > 1 ? 's' : ''} found · ${named} recognised`;
  // annotated image
  const wrap = $('#photo-wrap'); wrap.hidden = false;
  const cssW = wrap.clientWidth || 360, dpr = Math.min(devicePixelRatio || 1, 2), sc = (cssW * dpr) / W;
  const pc = $('#photo-canvas'); pc.width = Math.round(W * sc); pc.height = Math.round(H * sc);
  const g = pc.getContext('2d'); g.drawImage(c, 0, 0, pc.width, pc.height);
  const R = Math.max(10, 11 * dpr);
  res.forEach((r, i) => {
    const [x1, y1, x2, y2] = r.f.box.map((v) => v * sc), col = r.person ? known : grey;
    g.lineWidth = Math.max(2, 2 * dpr); g.strokeStyle = col; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const cx = Math.min(pc.width - R, Math.max(R, x1)), cy = Math.max(R, y1);
    g.fillStyle = col; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
    g.fillStyle = r.person ? '#1d1505' : '#0e1116'; g.font = `700 ${Math.round(R * 1.1)}px ui-monospace, monospace`;
    g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(i + 1), cx, cy + 1);
  });
  // result rows
  const list = $('#photo-results'); list.innerHTML = '';
  res.forEach((r, i) => {
    const b = document.createElement('button'); b.className = 'res' + (r.person ? ' known' : '');
    const small = r.f.box[2] - r.f.box[0] < 40;
    const others = r.top.filter((t) => t.p !== r.person).slice(0, 2).map((t) => `${esc(t.p.name)} ${t.s.toFixed(2)}`).join(' · ');
    let n, h, alt = '', sc2 = '';
    if (r.person) {
      n = esc(r.person.name); h = esc(r.person.headline || r.person.organisation || ''); sc2 = r.score.toFixed(2);
      if (others) alt = `Next closest: ${others}`;
    } else if (!app.people.length) { n = 'Face'; h = 'Import your face pack to name people'; }
    else {
      n = 'Unknown'; const t = r.top[0];
      h = t ? `Closest: <b>${esc(t.p.name)}</b> ${t.s.toFixed(2)}` : '';
      alt = small ? 'Face is small in this photo, so the result is less reliable' : 'Tap to add them, or save to someone in your list';
      sc2 = t ? t.s.toFixed(2) : '';
    }
    if (r.person && small) alt = 'Face is small in this photo, so double-check';
    b.innerHTML = `<span class="num">${i + 1}</span><img src="${r.crop}" alt=""><span class="t"><span class="n">${n}</span>${h ? `<span class="h">${h}</span>` : ''}${alt ? `<span class="alt">${alt}</span>` : ''}</span><span class="sc">${sc2}</span>`;
    b.onclick = () => r.person ? openPerson(r.person, r.score, true, { thumb: r.crop, emb: Float32Array.from(r.e), key: r.key }) : openAdd({ thumb: r.crop, emb: Float32Array.from(r.e), key: r.key });
    list.append(b);
  });
  $('#btn-pick-photo').textContent = res.length ? 'Check another photo' : 'Choose photo';
}

$('#btn-photo').onclick = () => { if (!app.photo) { $('#photo-sub').textContent = 'Pick a photo from your gallery. Group photos work too.'; } openSheet('#sheet-photo'); };
$('#btn-pick-photo').onclick = () => $('#file-check').click();
$('#file-check').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) checkPhoto(f); };

// photos shared to RTPI from other apps (Android share menu, installed app only)
async function takeSharedPhoto() {
  if (!new URLSearchParams(location.search).has('shared')) return;
  history.replaceState(null, '', location.pathname);
  try {
    const cache = await caches.open('rtpi-share'); const hit = await cache.match('shared-image');
    if (hit) { await cache.delete('shared-image'); await checkPhoto(await hit.blob()); }
  } catch (e) { console.error(e); toast("Couldn't open the shared photo."); }
}

/* fill in fingerprints for the active model from each person's saved face picture
   (for people added on this phone, or packs made before both models existed) */
async function backfill() {
  if (app.backfillRunning || !app.rec) return;
  const key = embKey(), model = app.recModel;
  const todo = [];
  for (const p of app.people) {
    if (!okEmb(p[key]) && p.thumb) todo.push({ p, obj: p });
    for (const x of p.extra || []) if (!okEmb(x[key]) && x.thumb) todo.push({ p, obj: x });
  }
  if (!todo.length) return;
  app.backfillRunning = true; app.backfill = { done: 0, total: todo.length };
  for (const { p, obj } of todo) {
    if (app.recModel !== model) break;
    try {
      const img = new Image(); img.src = obj.thumb; await img.decode();
      const c = canvas(img.naturalWidth, img.naturalHeight); c.getContext('2d').drawImage(img, 0, 0);
      const faces = await detect(c, c.width, c.height, 320, 0.4);
      if (faces.length) {
        const f0 = faces.sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
        while (app.rec.pending > 0) await sleep(30); // let the live camera go first
        const [e] = await embedMany(c, [f0]);
        obj[key] = e; await DB.putMany([p]).catch(() => {});
      }
    } catch (e) { console.warn('backfill', p.name, e); }
    app.backfill.done++; setStatus();
  }
  app.backfillRunning = false; app.backfill = null;
  rebuildIndex(); setStatus();
}

/* settings */
function syncSettings() {
  const s = app.settings;
  $('#s-model').value = app.recModel || s.model;
  $('#s-thr').value = thr(); $('#v-thr').textContent = thr().toFixed(2); $('#v-margin').textContent = (+s.margin).toFixed(2);
  $('#model-note').textContent = (app.recModel || s.model) === 'accurate'
    ? 'Best with turned heads and lookalikes. Names appear about a second after a face shows up.'
    : 'Names appear instantly, but turned heads and lookalikes are more often missed or confused.';
  $('#s-max').closest('label').hidden = (app.recModel || s.model) !== 'fast';
  save('rtpi-settings', s);
}
function bindSettings() {
  const s = app.settings;
  const sync = syncSettings;
  $('#s-margin').value = s.margin; $('#s-det').value = s.det; $('#s-max').value = s.max; $('#s-scores').checked = s.scores;
  $('#s-model').onchange = (e) => switchModel(e.target.value);
  $('#s-thr').oninput = (e) => { s['thr_' + (app.recModel || s.model)] = +e.target.value; sync(); };
  $('#s-margin').oninput = (e) => { s.margin = +e.target.value; sync(); };
  $('#s-det').onchange = (e) => { s.det = +e.target.value; sync(); };
  $('#s-max').onchange = (e) => { s.max = +e.target.value; sync(); };
  $('#s-scores').onchange = (e) => { s.scores = e.target.checked; sync(); };
  sync();
}
$('#btn-settings').onclick = () => openSheet('#sheet-settings');
$('#btn-flip').onclick = async () => { app.settings.facing = app.settings.facing === 'user' ? 'environment' : 'user'; save('rtpi-settings', app.settings); await startCamera(); };
document.addEventListener('visibilitychange', async () => {
  if (!document.hidden && app.stream && !app.wake) { try { app.wake = await navigator.wakeLock?.request('screen'); } catch { } }
  if (document.hidden) app.wake = null;
});

/* ---------- boot ---------- */
(async function boot() {
  sizeOverlay(); bindSettings();
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname))) navigator.serviceWorker.register('sw.js').catch(() => {});
  try { await DB.open(); } catch (e) { console.warn('IndexedDB unavailable; people will not be kept between visits', e); }
  showNotice('Loading face models', `About ${app.settings.model === 'accurate' ? 57 : 27} MB, downloaded once and then kept on this phone.`, [], true);
  try { await loadModels(); }
  catch (e) { console.error(e); app.loadError = true; showNotice("Couldn't load the face models", e.message || String(e), [{ label: 'Reload', primary: true, fn: () => location.reload() }]); return; }
  await reloadPeople();
  window.__rtpi = app;
  if (new URLSearchParams(location.search).has('shared')) { hideNotice(); await takeSharedPhoto(); }
  showNotice('Starting camera', 'Allow camera access when your browser asks.');
  if (await startCamera()) { updateNotice(); started = true; loop(); }
  syncSettings(); backfill();
  window.__rtpi = app; // for testing
})();
