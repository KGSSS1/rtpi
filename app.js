/* RTPI — real-time person identification, fully on-device.
   Pipeline: SCRFD-500M face detection (main thread, every frame) -> 5-point similarity alignment (ArcFace 112x112)
   -> 512-d embedding in a background worker (Accurate: ResNet-50 8-bit, Fast: MobileFaceNet)
   -> cosine match against the local people list. */
'use strict';

const APP_V = '7';
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
const DEFAULTS = { model: 'accurate', thr_accurate: 0.40, thr_fast: 0.42, margin: 0.05, det: 384, max: 3, scores: false, facing: 'user', maybe: true, learn: false, scope: '' };
const MAYBE_BAND = 0.05;   // how far below the threshold a match is still shown as "Possibly …"
const SHARP_MIN = 12;      // below this the aligned face is too blurred to trust (variance of Laplacian)
const SHARP_GOOD = 40;     // sharp enough to keep as a learned photo
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
  addDraft: null,
  hist: [], lastLog: new Map(), lastLearn: new Map(),
  indexV: 0, look: { v: -1, map: new Map(), busy: null },
  bfGen: 0, bfBusy: false, bfFailed: new WeakSet(),
  pFilter: '', pTab: 'people', card: null,
};

if (!MODELS[app.settings.model]) app.settings.model = 'accurate';
app.fb = load('rtpi-fb', { accurate: [], fast: [] });   // answers to "Is this …?": [score, 1 right | 0 wrong]

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
      const r = indexedDB.open('rtpi', 2);
      r.onupgradeneeded = () => {
        const d = r.result;
        if (!d.objectStoreNames.contains('people')) d.createObjectStore('people', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('history')) d.createObjectStore('history', { keyPath: 'id', autoIncrement: true });
      };
      r.onsuccess = () => { this.db = r.result; res(); }; r.onerror = () => rej(r.error);
    });
  },
  tx(mode, fn, store = 'people') {
    return new Promise((res, rej) => {
      const t = this.db.transaction(store, mode); const s = t.objectStore(store); const out = fn(s);
      t.oncomplete = () => res(out && out.result); t.onerror = () => rej(t.error);
    });
  },
  all() { return this.tx('readonly', (s) => s.getAll()); },
  putMany(arr) { return this.tx('readwrite', (s) => arr.forEach((p) => s.put(p))); },
  del(id) { return this.tx('readwrite', (s) => s.delete(id)); },
  clear() { return this.tx('readwrite', (s) => s.clear()); },
  histAll() { return this.tx('readonly', (s) => s.getAll(), 'history'); },
  histAdd(row) { return this.tx('readwrite', (s) => s.add(row), 'history'); },
  histDel(ids) { return this.tx('readwrite', (s) => ids.forEach((id) => s.delete(id)), 'history'); },
  histClear() { return this.tx('readwrite', (s) => s.clear(), 'history'); },
};

// matching index for the active model; people without a fingerprint for it are filled in by backfill()
const okEmb = (v) => v && v.length === DIM;
const inScope = (p) => !app.settings.scope || (p.tags || []).includes(app.settings.scope);
function rebuildIndex(resetTracks = true) {
  const key = embKey(), rows = [];
  for (const p of app.people) {
    if (!inScope(p)) continue;
    if (okEmb(p[key])) rows.push([p, p[key]]);
    for (const x of p.extra || []) if (okEmb(x[key])) rows.push([p, x[key]]);
  }
  app.rowP = rows.map((r) => r[0]);
  app.mPeople = [...new Set(app.rowP)];
  const E = new Float32Array(rows.length * DIM);
  rows.forEach(([, v], i) => E.set(v instanceof Float32Array ? v : new Float32Array(v), i * DIM));
  app.E = E; app.indexV++;
  if (resetTracks) app.tracks.forEach((t) => { t.emb = null; t.result = null; t.embAt = 0; });
}

function allTags() {
  const m = new Map();
  for (const p of app.people) for (const t of p.tags || []) m.set(t, (m.get(t) || 0) + 1);
  return [...m].sort((a, b) => a[0].localeCompare(b[0]));
}

async function reloadPeople() {
  let rows = [];
  try { rows = (await DB.all()) || []; } catch (e) { console.warn('storage unavailable', e); rows = app.people; }
  rows.sort((a, b) => a.name.localeCompare(b.name));
  app.people = rows;
  if (app.settings.scope && !allTags().some(([t]) => t === app.settings.scope)) { app.settings.scope = ''; save('rtpi-settings', app.settings); }
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
      close() { w.terminate(); waiting.forEach((j) => j.rej(new Error('recogniser closed'))); waiting.clear(); rec.pending = 0; },
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

// variance of the Laplacian over the centre of the aligned face (grey levels 0-255): low = blurred
function sharpness(px) {
  const g = new Float32Array(64 * 64);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) { const i = ((y + 24) * 112 + x + 24) * 4; g[y * 64 + x] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]; }
  let sum = 0, sq = 0, n = 0;
  for (let y = 1; y < 63; y++) for (let x = 1; x < 63; x++) { const c = y * 64 + x, v = 4 * g[c] - g[c - 1] - g[c + 1] - g[c - 64] - g[c + 64]; sum += v; sq += v * v; n++; }
  return sq / n - (sum / n) * (sum / n);
}

function alignedBatch(src, faces) {
  const n = 112 * 112, x = new Float32Array(faces.length * 3 * n);
  faces.forEach((f, k) => {
    const px = alignedPixels(src, f.pts), o = k * 3 * n; f.sharp = sharpness(px);
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

// sure: over the threshold and clearly ahead of the runner-up.
// maybe: just under the threshold, or over it without a clear lead -> shown as "Possibly …"
function classify(top) {
  const [a, b] = top;
  if (!a) return { person: null, maybe: null, closest: null, score: 0, second: 0 };
  const lead = a.s - Math.max(b ? b.s : 0, 0), t = thr();
  const sure = a.s >= t && lead >= app.settings.margin;
  const maybe = !sure && app.settings.maybe && a.s >= t - MAYBE_BAND && lead >= Math.min(app.settings.margin, 0.03);
  return { person: sure ? a.p : null, maybe: maybe ? a.p : null, closest: a.p, score: a.s, second: b ? b.s : 0 };
}
function match(e, deny) {
  let top = matchTop(e, deny ? 3 + deny.size : 3);
  if (deny) top = top.filter((t) => !deny.has(t.p.id));
  return classify(top);
}

/* ---------- feedback, calibration, history ---------- */
function recordFb(score, ok) {
  const m = app.recModel || app.settings.model;
  app.fb[m] = [...(app.fb[m] || []), [+(+score).toFixed(3), ok ? 1 : 0]].slice(-500);
  save('rtpi-fb', app.fb); renderCalib();
}
// threshold that would have made the fewest mistakes on the answers so far (a wrong name counts double a miss)
function calib(model) {
  const a = app.fb[model] || [], G = a.filter((x) => x[1]).map((x) => x[0]), I = a.filter((x) => !x[1]).map((x) => x[0]);
  const out = { n: a.length, g: G.length, i: I.length, thr: null };
  if (G.length < 10 || I.length < 3) return out;
  let best = Infinity, lo = 0, hi = 0, open = false;
  for (let k = 25; k <= 65; k++) {
    const t = k / 100, cost = G.filter((x) => x < t).length + 2 * I.filter((x) => x >= t).length;
    if (cost < best) { best = cost; lo = hi = t; open = true; } else if (cost === best && open) hi = t; else open = false;
  }
  out.thr = Math.round(((lo + hi) / 2) * 100) / 100;
  out.missed = G.filter((x) => x < out.thr).length; out.wrong = I.filter((x) => x >= out.thr).length;
  return out;
}
function renderCalib() {
  const m = app.recModel || app.settings.model, c = calib(m), el = $('#calib-text'); if (!el) return;
  $('#calib-val').textContent = c.thr != null ? c.thr.toFixed(2) : '';
  $('#calib-apply').hidden = c.thr == null || Math.abs(c.thr - thr()) < 0.005; $('#calib-reset').hidden = !c.n;
  if (!c.n) el.textContent = 'Answer "Is this …?" on results and the app works out the best threshold for your list.';
  else if (c.thr == null) el.textContent = `${c.n} answers so far (${c.g} right, ${c.i} wrong or not in your list). A suggestion needs at least 10 right and 3 wrong.`;
  else el.textContent = `From ${c.n} answers: at ${c.thr.toFixed(2)} the app would have missed ${c.missed} of ${c.g} right matches and let through ${c.wrong} of ${c.i} wrong ones. You are using ${thr().toFixed(2)}.`;
}

function logSeen(p, score, src, tier) {
  const now = Date.now();
  if (src === 'live') { if (now - (app.lastLog.get(p.id) || 0) < 5 * 60e3) return; app.lastLog.set(p.id, now); }
  const row = { t: now, pid: p.id, name: p.name, score: +score.toFixed(3), model: app.recModel, src, tier };
  app.hist.push(row);
  DB.histAdd(row).catch(() => {});
  if (openSheetEl === $('#sheet-people') && app.pTab === 'history') renderHistory();
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
    const t = bestT || { id: app.nextTrackId++, emb: null, seen: 0, embAt: 0, busy: false, firstSeen: performance.now(), hint: '' };
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
  if (!t.emb || same < 0.4) { t.emb = Float32Array.from(e); t.seen = 0; t.deny = null; t.confirmed = null; } // a different face in this spot: start fresh
  else { const w = Math.min(t.seen, 4); for (let d = 0; d < DIM; d++) t.emb[d] = t.emb[d] * w + e[d]; normalize(t.emb); }
  t.seen++; t.raw = e;
  const r = match(t.emb, t.deny);
  if (t.confirmed && (r.maybe === t.confirmed || r.person === t.confirmed)) { r.person = t.confirmed; r.maybe = null; } // you said yes to a "Possibly"
  t.result = r;
  if (r.person && t.seen >= 2) { logSeen(r.person, r.score, 'live', 'known'); learn(t); }
}

// Self-learning (off unless enabled): keep a very confident, sharp, new-looking view as an extra photo. Up to 5 per person.
function learn(t) {
  if (!app.settings.learn) return;
  const r = t.result, p = r.person, key = embKey();
  if (!p || t.seen < 3 || !(t.sharp >= SHARP_GOOD) || quickIssue(t)) return;
  if (r.score < Math.max(0.6, thr() + 0.2) || r.score - r.second < 0.2) return;
  if ((p.extra || []).filter((x) => String(x.id).startsWith('auto:')).length >= 5) return;
  if (Date.now() - (app.lastLearn.get(p.id) || 0) < 10 * 60e3) return;
  let mx = 0;
  for (const v of [p[key], ...(p.extra || []).map((x) => x[key])]) if (okEmb(v)) { let s = 0; for (let d = 0; d < DIM; d++) s += v[d] * t.raw[d]; if (s > mx) mx = s; }
  if (mx > 0.8 || mx < 0.55) return;   // nothing new to learn, or this single frame is not convincing on its own
  const snap = snapshotTrack(t); if (!snap) return;
  app.lastLearn.set(p.id, Date.now());
  const x = { id: 'auto:' + Date.now().toString(36), thumb: snap.thumb, added: new Date().toISOString().slice(0, 10) }; x[key] = snap.emb;
  p.extra = [...(p.extra || []), x];
  DB.putMany([p]).catch(() => {});
  rebuildIndex(false);
}

// quick checks that need no model: is the face big enough and roughly facing the camera?
function quickIssue(t) {
  if (t.box[2] - t.box[0] < 56) return 'Move closer';
  const [a, b, n] = t.pts, dx = b[0] - a[0], dy = b[1] - a[1], ed2 = dx * dx + dy * dy || 1;
  const off = Math.abs(((n[0] - (a[0] + b[0]) / 2) * dx + (n[1] - (a[1] + b[1]) / 2) * dy) / ed2);
  return off > 0.42 ? 'Face the camera' : '';
}

// Accurate: one face at a time, newest/unnamed first, each refreshed every ~1.5 s.
// Fast: every face (up to the setting) on every frame.
// Poor frames (small, turned or blurred faces) are skipped while a better one is likely to come:
// up to 1.2 s for a new face, up to 6 s for a face that already has a name.
function scheduleRecognition() {
  if (!app.rec || app.rec.pending > 0) return;
  const live = app.tracks.filter((t) => !t.missed && !t.busy);
  if (!live.length) return;
  const now = performance.now(); let cand, limit;
  if (app.recModel === 'fast') { cand = live; limit = +app.settings.max; }
  else {
    cand = live.filter((t) => !t.emb || now - t.embAt > 1500);
    cand.sort((a, b) => (a.emb ? 1 : 0) - (b.emb ? 1 : 0) || a.embAt - b.embAt); limit = 1;
  }
  const canWait = (t) => (t.emb ? now - t.embAt < 6000 : now - t.firstSeen < 1200);
  const pick = [];
  for (const t of cand) {
    const issue = quickIssue(t);
    if (issue && canWait(t)) { t.hint = issue; continue; }
    pick.push(t); if (pick.length >= limit) break;
  }
  if (!pick.length) return;
  const all = alignedBatch(frameCanvas, pick), n = 3 * 112 * 112;
  const keep = pick.filter((t) => !(t.sharp < SHARP_MIN && canWait(t)));
  pick.forEach((t) => { t.hint = keep.includes(t) ? '' : 'Hold still'; });
  if (!keep.length) return;
  let x = all;
  if (keep.length !== pick.length) { x = new Float32Array(keep.length * n); keep.forEach((t, i) => x.set(all.subarray(pick.indexOf(t) * n, (pick.indexOf(t) + 1) * n), i * n)); }
  keep.forEach((t) => { t.busy = true; });
  const model = app.recModel;
  app.rec.embed(x, keep.length).then((embs) => {
    keep.forEach((t, i) => { t.busy = false; t.embAt = performance.now(); if (model === app.recModel) applyEmbedding(t, embs[i]); });
    draw();
  }).catch((e) => { keep.forEach((t) => (t.busy = false)); if (!/closed/.test(e.message)) console.error(e); });
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
    const r = t.result || {}; const isKnown = !!r.person, isMaybe = !isKnown && !!r.maybe; const col = isKnown || isMaybe ? known : unknown;
    // corner brackets (dashed for a "Possibly")
    const L = Math.max(12, Math.min(w, h) * 0.22);
    octx.strokeStyle = col; octx.lineWidth = isKnown ? 3 : 2; octx.lineCap = 'round'; octx.setLineDash(isMaybe ? [5, 6] : []); octx.beginPath();
    [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]].forEach(([x, y, dx, dy]) => { octx.moveTo(x + dx * L, y); octx.lineTo(x, y); octx.lineTo(x, y + dy * L); });
    octx.stroke(); octx.setLineDash([]);
    // name tag
    const tagW = Math.max(150, Math.min(280, w * 1.4)), pad = 10;
    let title, line2;
    if (isKnown) { title = r.person.name; line2 = r.person.headline || r.person.organisation || ''; }
    else if (isMaybe) { title = 'Possibly ' + r.maybe.name; line2 = 'Tap to confirm'; }
    else if (!app.people.length) { title = 'Face found'; line2 = 'Import your face pack to name people'; }
    else if (!t.result) { title = t.hint ? t.hint + '…' : 'Identifying…'; line2 = ''; }
    else { title = 'Unknown'; line2 = app.settings.scores && r.closest ? `closest: ${r.closest.name}` : 'Tap to add'; }
    if (app.settings.scores && r.closest && t.result) title += `  ${r.score.toFixed(2)}`;
    octx.font = '700 15px ' + css.getPropertyValue('--ui'); const t1 = fit(title, tagW - pad * 2);
    octx.font = '13px ' + css.getPropertyValue('--ui'); const t2 = line2 ? fit(line2, tagW - pad * 2) : '';
    const tagH = t2 ? 46 : 28;
    let tx = (x1 + x2) / 2 - tagW / 2; tx = Math.max(8, Math.min(cw - tagW - 8, tx));
    let ty = y2 + 10; if (ty + tagH > ch - 110) ty = Math.max(96, y1 - tagH - 10);
    octx.fillStyle = isKnown ? known : 'rgba(23,27,34,.86)';
    octx.beginPath(); octx.roundRect(tx, ty, tagW, tagH, 10); octx.fill();
    if (isMaybe) { octx.strokeStyle = known; octx.lineWidth = 1; octx.stroke(); }
    octx.fillStyle = isKnown ? css.getPropertyValue('--known-ink') : isMaybe ? known : css.getPropertyValue('--text');
    octx.font = '700 15px ' + css.getPropertyValue('--ui'); octx.textBaseline = 'top'; octx.fillText(t1, tx + pad, ty + 7);
    octx.fillStyle = isKnown ? css.getPropertyValue('--known-ink') : css.getPropertyValue('--text');
    if (t2) { octx.globalAlpha = 0.8; octx.font = '13px ' + css.getPropertyValue('--ui'); octx.fillText(t2, tx + pad, ty + 26); octx.globalAlpha = 1; }
    app.hits.push({ rect: [Math.min(x1, tx), Math.min(y1, ty), Math.max(x2, tx + tagW), Math.max(y2, ty + tagH)], track: t });
  }
}

overlay.addEventListener('click', (ev) => {
  const h = app.hits.find(({ rect: [a, b, c, d] }) => ev.clientX >= a && ev.clientX <= c && ev.clientY >= b && ev.clientY <= d);
  if (!h) return;
  const r = h.track.result || {};
  if (r.person) openPerson(r.person, { score: r.score, face: snapshotTrack(h.track), tier: 'known', track: h.track });
  else if (r.maybe) openPerson(r.maybe, { score: r.score, face: snapshotTrack(h.track), tier: 'maybe', track: h.track });
  else openAdd(snapshotTrack(h.track));
});

/* ---------- status + notice ---------- */
function setStatus() {
  const m = app.recModel || app.settings.model, sc = app.settings.scope;
  $('#status').innerHTML = `<b>${sc ? app.mPeople.length : app.people.length}</b>${sc ? ' ' + esc(sc) : ''} · ${MODELS[m].name} · ${app.fps.toFixed(0)} fps`;
}
// progress of preparing fingerprints, on its own line under the top bar
function showPrep() {
  const b = app.backfill, el = $('#prep'); el.hidden = !b; if (!b) return;
  $('#prep-text').textContent = `Preparing ${MODELS[b.model].name} model: ${b.done} of ${b.total} photos`;
  $('#prep-bar').style.width = (b.total ? (b.done / b.total) * 100 : 0).toFixed(1) + '%';
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

function openPerson(p, o = {}) {
  const { score = null, fromPhoto = false, face = null, tier = 'known', track = null, row = null } = o;
  app.card = { p, score, fromPhoto, tier, track, row, answered: false };
  app.cardPerson = p; app.cardFace = face && okEmb(face.emb) ? face : null;
  renderCard();
  openSheet('#sheet-person');
}

function renderCard() {
  const { p, score, fromPhoto, tier, answered } = app.card, t0 = thr();
  const n = 1 + (p.extra || []).length, first = p.name.split(' ')[0];
  $('#person-body').innerHTML = `${fromPhoto ? '<button class="btn back" id="btn-back-photo">← Back to photo</button>' : ''}
    <div class="person">
      <img src="${esc(p.thumb)}" alt="">
      <div class="who">
        <div class="name">${esc(p.name)}${p.source === 'added' ? '<span class="tag">added</span>' : ''}</div>
        ${p.headline ? `<p class="head">${esc(p.headline)}</p>` : ''}
        <div class="meta">${[p.organisation, p.location].filter(Boolean).map(esc).join(' · ')}</div>
      </div>
    </div>
    ${(p.tags || []).length ? `<div class="chips">${p.tags.map((t) => `<span class="chipb static">${esc(t)}</span>`).join('')}</div>` : ''}
    ${p.notes ? `<p class="notes">${esc(p.notes)}</p>` : ''}
    ${score != null ? `<div class="match"><span>${tier === 'maybe' ? 'possible' : 'match'}</span><div class="meter"><div style="width:${Math.max(0, Math.min(1, score)) * 100}%"></div><i style="left:${t0 * 100}%"></i></div><span>${score.toFixed(2)}</span></div>
    <div class="fb" id="fb-row">${answered ? '<span>Answer recorded.</span>' : `<span>Is this ${esc(first)}?</span><button class="btn" id="fb-yes">Yes</button><button class="btn" id="fb-no">No</button>`}</div>` : ''}
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
    <div class="row">
      <button class="btn" id="btn-edit-person">Edit details, tags and notes</button>
      <button class="btn danger" id="btn-del-person">Remove from this phone</button>
    </div>`;
  renderStrip(p);
  if ($('#fb-yes')) { $('#fb-yes').onclick = () => answer(true); $('#fb-no').onclick = () => answer(false); }
  if ($('#btn-save-face')) $('#btn-save-face').onclick = () => addExtra(p, app.cardFace);
  $('#btn-add-extra').onclick = () => $('#file-extra').click();
  $('#btn-edit-person').onclick = () => editPerson(p);
  if (fromPhoto) $('#btn-back-photo').onclick = () => { renderPhoto(); openSheet('#sheet-photo'); };
  armButton($('#btn-del-person'), 'Tap again to remove', async () => { await DB.del(p.id); await reloadPeople(); closeSheet(); toast(`Removed ${p.name}`); });
}

// "Is this …?" — every answer feeds the threshold calibration
function answer(ok) {
  const c = app.card; if (!c || c.answered || c.score == null) return;
  c.answered = true; recordFb(c.score, ok);
  const row = $('#fb-row'); if (row) row.innerHTML = `<span>${ok ? 'Noted as right.' : 'Noted as wrong. This face will not be named as them again here.'}</span>`;
  if (ok) {
    if (c.tier === 'maybe') { if (c.track) { c.track.confirmed = c.p; if (c.track.result) { c.track.result.person = c.p; c.track.result.maybe = null; } } if (c.row) { c.row.person = c.p; c.row.maybe = null; } logSeen(c.p, c.score, c.row ? 'photo' : 'live', 'confirmed'); }
  } else {
    if (c.track) { (c.track.deny ||= new Set()).add(c.p.id); c.track.confirmed = null; if (c.track.emb) c.track.result = match(c.track.emb, c.track.deny); }
    if (c.row) { c.row.person = null; c.row.maybe = null; c.row.top = c.row.top.filter((t) => t.p !== c.p); }
    app.cardFace = null; $('#btn-save-face')?.remove();
  }
}

function renderStrip(p) {
  const strip = $('#photo-strip'); if (!strip) return; strip.innerHTML = '';
  const main = document.createElement('figure'); main.innerHTML = `<img src="${esc(p.thumb)}" alt=""><figcaption>Main</figcaption>`; strip.append(main);
  (p.extra || []).forEach((x) => {
    const f = document.createElement('figure');
    f.innerHTML = `<img src="${esc(x.thumb)}" alt="">${String(x.id).startsWith('auto:') ? '<figcaption class="top">learned</figcaption>' : ''}<button class="rm" aria-label="Remove this photo">×</button><button class="mk" aria-label="Use as main photo">Main</button>`;
    armButton(f.querySelector('.rm'), '?', async () => {
      p.extra = p.extra.filter((y) => y !== x); await DB.putMany([p]).catch(() => {});
      rebuildIndex(false); renderStrip(p); toast('Photo removed');
    });
    f.querySelector('.mk').onclick = () => makeMain(p, x);
    strip.append(f);
  });
  const cnt = $('.photos .count'); if (cnt) cnt.textContent = 1 + (p.extra || []).length;
}

// swap an extra photo with the main one (the picture shown everywhere for this person)
async function makeMain(p, x) {
  const old = { id: 'ph:' + Date.now().toString(36), thumb: p.thumb, added: '' };
  for (const k of ['emb', 'emb_r50']) { if (okEmb(p[k])) old[k] = p[k]; if (okEmb(x[k])) p[k] = x[k]; else delete p[k]; }
  p.thumb = x.thumb; p.keepPhoto = true; delete p.face_px;
  p.extra = p.extra.map((y) => (y === x ? old : y));
  await DB.putMany([p]).catch(() => {});
  rebuildIndex(false); renderCard(); renderPeopleList(); toast('Main photo changed'); backfill();
}

// add another photo (template) to a person
async function addExtra(p, face) {
  const key = face.key || embKey();
  const before = matchTop(face.emb, 50).find((t) => t.p === p);
  const x = { id: 'ph:' + Date.now().toString(36), thumb: face.thumb, added: new Date().toISOString().slice(0, 10) };
  x[key] = face.emb;
  p.extra = [...(p.extra || []), x];
  await DB.putMany([p]).catch(() => {});
  rebuildIndex(false);
  const c = app.card;
  if (c && c.p === p && app.cardFace === face) {
    if (!c.answered && c.score != null) answer(true);   // saving the face means the name was right
    app.cardFace = null; $('#btn-save-face')?.remove();
  }
  renderStrip(p);
  if (before && before.s < 0.2) toast(`Saved, but this face looks quite different from ${p.name}'s other photos (${before.s.toFixed(2)}). If it's someone else, remove it.`, 5000);
  else toast(`Saved. ${p.name} now has ${1 + p.extra.length} photos.`);
  backfill();
}

// edit details, tags and notes on the phone; edited details are kept when a face pack is imported later
function editPerson(p) {
  $('#person-body').innerHTML = `<h3>Edit details</h3>
    <form class="stack" id="edit-form" autocomplete="off">
      <label class="field">Name<input type="text" id="e-name" required></label>
      <label class="field">Headline<input type="text" id="e-headline"></label>
      <label class="field">College / company<input type="text" id="e-org"></label>
      <label class="field">Location<input type="text" id="e-location"></label>
      <label class="field">Profile link<input type="url" id="e-link" placeholder="https://"></label>
      <label class="field">Tags, separated by commas<input type="text" id="e-tags" placeholder="e.g. Section B, Finance Club"></label>
      <label class="field">Notes<textarea id="e-notes" rows="3" placeholder="Where you met, what to remember"></textarea></label>
      <div class="row"><button class="btn" type="button" id="e-cancel">Cancel</button><button class="btn primary" type="submit">Save</button></div>
    </form>`;
  const f = { name: '#e-name', headline: '#e-headline', organisation: '#e-org', location: '#e-location', link: '#e-link' };
  for (const k in f) $(f[k]).value = p[k] || '';
  $('#e-tags').value = (p.tags || []).join(', '); $('#e-notes').value = p.notes || '';
  $('#e-cancel').onclick = renderCard;
  $('#edit-form').onsubmit = async (e) => {
    e.preventDefault();
    const name = $('#e-name').value.trim(); if (!name) return;
    let changed = false;
    for (const k in f) { const v = $(f[k]).value.trim(); if (v !== (p[k] || '')) { p[k] = v; changed = true; } }
    if (changed) p.edited = true;
    p.tags = [...new Set($('#e-tags').value.split(',').map((t) => t.trim()).filter(Boolean))];
    p.notes = $('#e-notes').value.trim();
    await DB.putMany([p]).catch(() => {});
    app.people.sort((a, b) => a.name.localeCompare(b.name));
    if (app.settings.scope && !allTags().some(([t]) => t === app.settings.scope)) { app.settings.scope = ''; save('rtpi-settings', app.settings); }
    rebuildIndex(false); renderCard(); renderPeopleList(); toast('Saved');
  };
}

function armButton(btn, armedLabel, fn) {
  const label = btn.textContent; let armed = false, t;
  btn.onclick = () => {
    if (armed) { clearTimeout(t); fn(); return; }
    armed = true; btn.classList.add('armed'); btn.textContent = armedLabel;
    t = setTimeout(() => { armed = false; btn.classList.remove('armed'); btn.textContent = label; }, 3500);
  };
}

/* people list: filters, needs-attention, tags */
// who needs a look, and why
function attention() {
  const key = embKey(), m = new Map(), add = (p, why) => m.set(p, [...(m.get(p) || []), why]);
  const avg = new Map();
  for (const h of app.hist) if (h.model === app.recModel && h.tier === 'known') { const a = avg.get(h.pid) || [0, 0]; a[0] += h.score; a[1]++; avg.set(h.pid, a); }
  const look = app.look.v === app.indexV ? app.look.map : null;
  for (const p of app.people) {
    if (!okEmb(p[key]) && (app.bfFailed.has(p) || !p.thumb)) add(p, 'No usable face in the main photo');   // ones still being prepared are not flagged
    if (p.face_px && p.face_px < 110) add(p, 'Small face in the main photo');
    const lk = look && look.get(p); if (lk) add(p, `Looks like ${lk.q.name} (${lk.s.toFixed(2)})`);
    const a = avg.get(p.id); if (a && a[1] >= 2 && a[0] / a[1] < thr() + 0.07) add(p, `Weak matches so far (average ${(a[0] / a[1]).toFixed(2)})`);
  }
  return m;
}
// pairs of different people whose photos look alike to the current model (done in small steps so the camera keeps running)
async function computeLookalikes() {
  const v = app.indexV; if (app.look.v === v || app.look.busy === v) return; app.look.busy = v;
  const E = app.E, P = app.rowP, R = P.length, map = new Map(), cut = thr() + 0.05;
  for (let i = 0; i < R; i++) {
    for (let j = i + 1; j < R; j++) {
      if (P[i] === P[j]) continue;
      let s = 0; const a = i * DIM, b = j * DIM; for (let d = 0; d < DIM; d++) s += E[a + d] * E[b + d];
      if (s >= cut) for (const [x, y] of [[P[i], P[j]], [P[j], P[i]]]) { const c = map.get(x); if (!c || s > c.s) map.set(x, { q: y, s }); }
    }
    if (i % 30 === 29) { await sleep(0); if (app.indexV !== v) { if (app.look.busy === v) app.look.busy = null; return; } }
  }
  app.look = { v, map, busy: null };
  if (openSheetEl === $('#sheet-people') && app.pTab === 'people') renderPeopleList();
}

function shownPeople(att) {
  const q = ($('#people-search').value || '').toLowerCase().trim(), f = app.pFilter;
  return app.people.filter((p) => (!q || [p.name, p.headline, p.organisation, p.location, p.notes, (p.tags || []).join(' ')].join(' ').toLowerCase().includes(q))
    && (!f || (f === 'attention' ? att.has(p) : (p.tags || []).includes(f.slice(4)))));
}

function renderPeopleList() {
  const tags = allTags(), att = attention();
  if (app.pFilter && (app.pFilter === 'attention' ? !att.size : !tags.some(([t]) => 'tag:' + t === app.pFilter))) app.pFilter = '';
  const chips = $('#people-chips'); chips.innerHTML = '';
  const chip = (label, val) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'chipb' + (app.pFilter === val ? ' on' : ''); b.textContent = label; b.onclick = () => { app.pFilter = val; renderPeopleList(); }; chips.append(b); };
  chip(`All ${app.people.length}`, ''); if (att.size) chip(`Needs attention ${att.size}`, 'attention'); tags.forEach(([t, n]) => chip(`${t} ${n}`, 'tag:' + t));
  chips.hidden = !app.people.length;
  const sel = $('#scope'); sel.innerHTML = '';
  [['', `Everyone (${app.people.length})`], ...tags.map(([t, n]) => [t, `Only "${t}" (${n})`])].forEach(([v, l]) => { const o = document.createElement('option'); o.value = v; o.textContent = l; sel.append(o); });
  sel.value = app.settings.scope || ''; $('#scope-field').hidden = !tags.length;
  const rows = shownPeople(att);
  $('#people-title').textContent = `People (${app.people.length})`;
  $('#bulk').hidden = !app.people.length; $('#bulk-apply').textContent = `Tag ${rows.length} shown`;
  const list = $('#people-list'); list.innerHTML = '';
  const frag = document.createDocumentFragment();
  rows.slice(0, 400).forEach((p) => {
    const b = document.createElement('button'); const why = att.get(p);
    const sub = app.pFilter === 'attention' && why ? why.join(' · ') : (p.headline || p.organisation || '');
    b.innerHTML = `<img loading="lazy" src="${esc(p.thumb)}" alt=""><span class="t"><div class="n">${esc(p.name)}${why && app.pFilter !== 'attention' ? '<i class="dot" title="Needs attention"></i>' : ''}</div><div class="h${app.pFilter === 'attention' ? ' warn' : ''}">${esc(sub)}</div></span>`;
    b.onclick = () => openPerson(p);
    frag.append(b);
  });
  list.append(frag);
  if (rows.length > 400) { const m = document.createElement('p'); m.className = 'sub'; m.textContent = `Showing the first 400 of ${rows.length}. Search to narrow it down.`; list.append(m); }
  if (!app.people.length) list.innerHTML = '<p class="sub">Nobody yet. Import face_pack.json, or add people one at a time with the + button.</p>';
  else if (!rows.length) list.innerHTML = '<p class="sub">No one matches.</p>';
}
$('#people-search').addEventListener('input', renderPeopleList);
$('#scope').onchange = (e) => { app.settings.scope = e.target.value; save('rtpi-settings', app.settings); rebuildIndex(); setStatus(); toast(e.target.value ? `Recognising only "${e.target.value}" (${app.mPeople.length} people)` : 'Recognising everyone'); };
$('#bulk-tag').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('#bulk-apply').click(); } });
$('#bulk-apply').onclick = async () => {
  let tag = $('#bulk-tag').value.trim(); if (!tag) { toast('Type a tag first. Start it with - to remove that tag instead.'); return; }
  const remove = tag.startsWith('-'); if (remove) tag = tag.slice(1).trim(); if (!tag) return;
  const rows = shownPeople(attention());
  rows.forEach((p) => { const s = new Set(p.tags || []); remove ? s.delete(tag) : s.add(tag); p.tags = [...s]; });
  await DB.putMany(rows).catch(() => {});
  $('#bulk-tag').value = '';
  if (app.settings.scope && !allTags().some(([t]) => t === app.settings.scope)) { app.settings.scope = ''; save('rtpi-settings', app.settings); }
  rebuildIndex(false); renderPeopleList(); toast(`${remove ? 'Removed' : 'Added'} "${tag}" ${remove ? 'from' : 'on'} ${rows.length} people`);
};

/* history: who was recognised and when (kept on this phone only) */
function renderHistory() {
  const q = ($('#hist-search').value || '').toLowerCase().trim(), byId = new Map(app.people.map((p) => [p.id, p]));
  const rows = app.hist.filter((h) => !q || h.name.toLowerCase().includes(q)).slice(-400).reverse();
  const list = $('#hist-list'); list.innerHTML = ''; let day = '';
  const frag = document.createDocumentFragment();
  rows.forEach((h) => {
    const d = new Date(h.t), label = d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' });
    if (label !== day) { day = label; const hd = document.createElement('div'); hd.className = 'day'; hd.textContent = label; frag.append(hd); }
    const p = byId.get(h.pid), b = document.createElement('button');
    b.innerHTML = `<img loading="lazy" src="${p ? esc(p.thumb) : ''}" alt=""><span class="t"><div class="n">${esc(p ? p.name : h.name)}</div><div class="h">${d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })} · ${h.src === 'photo' ? 'photo' : 'camera'}${h.tier === 'confirmed' ? ' · confirmed by you' : ''} · ${h.score.toFixed(2)}</div></span>`;
    b.onclick = () => (p ? openPerson(p) : toast(`${h.name} is no longer in your list.`));
    frag.append(b);
  });
  list.append(frag);
  if (!rows.length) list.innerHTML = `<p class="sub">${app.hist.length ? 'No one by that name.' : 'Nobody yet. People you recognise with the camera or in photos show up here.'}</p>`;
  $('#btn-clear-hist').hidden = !app.hist.length;
}
$('#hist-search').addEventListener('input', renderHistory);
armButton($('#btn-clear-hist'), 'Tap again to clear history', async () => { await DB.histClear().catch(() => {}); app.hist = []; app.lastLog.clear(); renderHistory(); toast('History cleared'); });
function setPeopleTab(tab) {
  app.pTab = tab;
  $('#tab-people').classList.toggle('on', tab === 'people'); $('#tab-history').classList.toggle('on', tab === 'history');
  $('#pane-people').hidden = tab !== 'people'; $('#pane-history').hidden = tab !== 'history';
  if (tab === 'history') renderHistory(); else { renderPeopleList(); computeLookalikes(); }
}
$('#tab-people').onclick = () => setPeopleTab('people');
$('#tab-history').onclick = () => setPeopleTab('history');
$('#btn-people').onclick = () => { setPeopleTab(app.pTab); openSheet('#sheet-people'); };

/* import and export */
$('#btn-import').onclick = () => $('#file-pack').click();
async function importPack(text) {
  const pack = JSON.parse(text);
  if (pack.format !== 'face-pack' || !Array.isArray(pack.people)) throw new Error('This file is not an RTPI face pack.');
  const have = new Map(app.people.map((p) => [p.id, p]));
  const embs = (src, name) => { const o = {}; for (const k of ['emb', 'emb_r50']) if (src[k]) { const v = b64ToF32(src[k]); if (v.length !== DIM) throw new Error(`Bad face data for ${name}.`); o[k] = v; } return o; };
  const rows = pack.people.map((p) => {
    const id = String(p.id), old = have.get(id) || {};
    const row = { ...old, id, source: old.source || p.source || 'pack' };
    // details changed on this phone win over the pack
    if (!old.edited) for (const k of ['name', 'headline', 'organisation', 'location', 'link']) row[k] = p[k] || '';
    const main = { thumb: p.thumb || '', ...embs(p, p.name) };
    if (!old.keepPhoto) {
      if (old.thumb && old.thumb !== main.thumb) for (const k of ['emb', 'emb_r50']) if (!main[k]) delete row[k];   // photo changed: stale fingerprint
      Object.assign(row, main);
      if (p.face_px) row.face_px = p.face_px; else delete row.face_px;
    }
    for (const k of ['tags', 'notes', 'edited', 'keepPhoto']) if (p[k] != null && old[k] == null) row[k] = p[k];
    // extra photos: ones from the computer (pc:) are replaced by the pack's, ones added on this phone are kept
    const ex = new Map((old.extra || []).map((x) => [x.id, x]));
    if (Array.isArray(p.extra)) {
      for (const idx of [...ex.keys()]) if (String(idx).startsWith('pc:') && idx !== 'pc:main') ex.delete(idx);
      p.extra.forEach((x) => { const xid = String(x.id || 'pc:' + Math.random().toString(36).slice(2)); ex.set(xid, { id: xid, thumb: x.thumb || '', added: x.added || '', ...embs(x, p.name) }); });
    }
    if (old.keepPhoto && main.thumb !== row.thumb) ex.set('pc:main', { id: 'pc:main', added: '', ...main });   // you chose another main photo: keep the pack's as an extra
    if (ex.size || old.extra) row.extra = [...ex.values()];
    return row;
  });
  await DB.putMany(rows).catch(() => { app.people = rows; });
  await reloadPeople();
  toast(`Imported ${rows.length} people`);
  backfill();
}
$('#file-pack').onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try { await importPack(await f.text()); } catch (err) { toast(err.message || 'Could not read that file', 4000); }
};
$('#btn-export').onclick = () => {
  const enc = (v) => v ? f32ToB64(v instanceof Float32Array ? v : new Float32Array(v)) : undefined;
  const pack = { format: 'face-pack', version: 3, models: ['w600k_mbf', 'w600k_r50'], dim: DIM, created: new Date().toISOString().slice(0, 16).replace('T', ' '),
    people: app.people.map((p) => ({ id: p.id, name: p.name, headline: p.headline, organisation: p.organisation, location: p.location, link: p.link, thumb: p.thumb, source: p.source,
      tags: p.tags, notes: p.notes, edited: p.edited, keepPhoto: p.keepPhoto, face_px: p.face_px, emb: enc(p.emb), emb_r50: enc(p.emb_r50),
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
      // the app said Unknown but they were in the list: a missed match, useful for calibration
      const d = app.addDraft, hit = d.key === embKey() ? matchTop(d.emb, 50).find((t) => t.p === p) : null;
      if (hit) recordFb(hit.s, true);
      await addExtra(p, d); app.addDraft = null; closeSheet();
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
  if (dup && dup.closest && app.addDraft.key === embKey()) recordFb(dup.score, false);   // a new person: their closest match was not them
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
  const res = faces.map((f, i) => ({ f, e: embs[i], key, top: matchTop(embs[i], 3), person: null, maybe: null, crop: cropData(c, f.box) }));
  const taken = new Set();
  [...res].sort((a, b) => (b.top[0]?.s || 0) - (a.top[0]?.s || 0)).forEach((r) => {
    const m = classify(r.top); r.score = m.score;
    if (m.person && !taken.has(m.person.id)) { r.person = m.person; taken.add(m.person.id); logSeen(m.person, m.score, 'photo', 'known'); }
    else if ((m.maybe || m.person) && !taken.has((m.maybe || m.person).id)) r.maybe = m.maybe || m.person;
  });
  app.photo = { c, W, H, res };
  renderPhoto();
}

function renderPhoto() {
  if (!app.photo) return;
  const { c, W, H, res } = app.photo;
  const known = getComputedStyle(document.documentElement).getPropertyValue('--known').trim();
  const grey = getComputedStyle(document.documentElement).getPropertyValue('--unknown').trim();
  const named = res.filter((r) => r.person).length, poss = res.filter((r) => !r.person && r.maybe).length;
  $('#photo-sub').textContent = !res.length ? 'No faces found. Try a clearer photo where faces are larger.'
    : !app.people.length ? `${res.length} face${res.length > 1 ? 's' : ''} found. Import your face pack to name them.`
    : `${res.length} face${res.length > 1 ? 's' : ''} found · ${named} recognised${poss ? ` · ${poss} possible` : ''}`;
  // annotated image
  const wrap = $('#photo-wrap'); wrap.hidden = false;
  const cssW = wrap.clientWidth || 360, dpr = Math.min(devicePixelRatio || 1, 2), sc = (cssW * dpr) / W;
  const pc = $('#photo-canvas'); pc.width = Math.round(W * sc); pc.height = Math.round(H * sc);
  const g = pc.getContext('2d'); g.drawImage(c, 0, 0, pc.width, pc.height);
  const R = Math.max(10, 11 * dpr);
  res.forEach((r, i) => {
    const [x1, y1, x2, y2] = r.f.box.map((v) => v * sc), col = r.person || r.maybe ? known : grey;
    g.lineWidth = Math.max(2, 2 * dpr); g.strokeStyle = col; g.setLineDash(!r.person && r.maybe ? [4 * dpr, 4 * dpr] : []); g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.setLineDash([]);
    const cx = Math.min(pc.width - R, Math.max(R, x1)), cy = Math.max(R, y1);
    g.fillStyle = r.person ? known : grey; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
    g.fillStyle = r.person ? '#1d1505' : '#0e1116'; g.font = `700 ${Math.round(R * 1.1)}px ui-monospace, monospace`;
    g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(i + 1), cx, cy + 1);
  });
  // result rows
  const list = $('#photo-results'); list.innerHTML = '';
  res.forEach((r, i) => {
    const who = r.person || r.maybe;
    const b = document.createElement('button'); b.className = 'res' + (r.person ? ' known' : r.maybe ? ' maybe' : '');
    const small = r.f.box[2] - r.f.box[0] < 40;
    const others = r.top.filter((t) => t.p !== who).slice(0, 2).map((t) => `${esc(t.p.name)} ${t.s.toFixed(2)}`).join(' · ');
    let n, h, alt = '', sc2 = '';
    if (who) {
      n = (r.person ? '' : 'Possibly ') + esc(who.name); h = esc(who.headline || who.organisation || ''); sc2 = r.score.toFixed(2);
      alt = r.person ? (others ? `Next closest: ${others}` : '') : 'Tap to confirm or reject';
    } else if (!app.people.length) { n = 'Face'; h = 'Import your face pack to name people'; }
    else {
      n = 'Unknown'; const t = r.top[0];
      h = t ? `Closest: <b>${esc(t.p.name)}</b> ${t.s.toFixed(2)}` : '';
      alt = small ? 'Face is small in this photo, so the result is less reliable' : 'Tap to add them, or save to someone in your list';
      sc2 = t ? t.s.toFixed(2) : '';
    }
    if (r.person && small) alt = 'Face is small in this photo, so double-check';
    b.innerHTML = `<span class="num">${i + 1}</span><img src="${r.crop}" alt=""><span class="t"><span class="n">${n}</span>${h ? `<span class="h">${h}</span>` : ''}${alt ? `<span class="alt">${alt}</span>` : ''}</span><span class="sc">${sc2}</span>`;
    const face = () => ({ thumb: r.crop, emb: Float32Array.from(r.e), key: r.key });
    b.onclick = () => (who ? openPerson(who, { score: r.score, fromPhoto: true, face: face(), tier: r.person ? 'known' : 'maybe', row: r }) : openAdd(face()));
    list.append(b);
  });
  $('#btn-pick-photo').textContent = res.length ? 'Check another photo' : 'Choose photo';
}

$('#btn-photo').onclick = () => { if (!app.photo) { $('#photo-sub').textContent = 'Pick a photo from your gallery. Group photos work too.'; } openSheet('#sheet-photo'); };
$('#btn-pick-photo').onclick = () => $('#file-check').click();
$('#file-check').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) checkPhoto(f); };

// a photo or a face pack shared to RTPI from another app (Android share menu, installed app only)
async function takeShared() {
  if (!new URLSearchParams(location.search).has('shared')) return;
  history.replaceState(null, '', location.pathname);
  try {
    const cache = await caches.open('rtpi-share');
    const pack = await cache.match('shared-pack');
    if (pack) {
      await cache.delete('shared-pack');
      try { await importPack(await pack.text()); } catch (err) { toast(err.message || 'Could not read that file', 4000); }
      return;
    }
    const hit = await cache.match('shared-image');
    if (hit) { await cache.delete('shared-image'); await checkPhoto(await hit.blob()); }
  } catch (e) { console.error(e); toast("Couldn't open the shared file."); }
}

/* Fingerprints are prepared in the background for BOTH models, whichever one is in use:
   first the active model, then the other one (only if its files are already on this phone).
   Sources are each person's saved face pictures, so this covers people added on the phone and older packs. */
function missingFor(key) {
  const todo = [];
  for (const p of app.people) {
    if (!okEmb(p[key]) && p.thumb && !app.bfFailed.has(p)) todo.push({ p, obj: p });
    for (const x of p.extra || []) if (!okEmb(x[key]) && x.thumb && !app.bfFailed.has(x)) todo.push({ p, obj: x });
  }
  return todo;
}
async function modelCached(model) {
  try { const c = await caches.open(ASSET_CACHE); for (const a of MODELS[model].parts) if (!(await c.match(a.url))) return false; return true; } catch { return false; }
}
async function fillModel(model, rec, gen, active) {
  const key = MODELS[model].key, todo = missingFor(key); if (!todo.length) return;
  app.backfill = { model, done: 0, total: todo.length }; showPrep(); let fresh = 0;
  for (const { p, obj } of todo) {
    if (gen !== app.bfGen) return;
    try {
      const img = new Image(); img.src = obj.thumb; await img.decode();
      const c = canvas(img.naturalWidth, img.naturalHeight); c.getContext('2d').drawImage(img, 0, 0);
      const faces = await detect(c, c.width, c.height, 320, 0.4);
      if (faces.length) {
        while (rec.pending > 0) await sleep(30);   // the live camera goes first
        if (gen !== app.bfGen) return;
        const [e] = await rec.embed(alignedBatch(c, [largest(faces)]), 1);
        obj[key] = e; await DB.putMany([p]).catch(() => {}); fresh++;
      } else app.bfFailed.add(obj);
    } catch (e) { if (/closed/.test(e.message)) return; app.bfFailed.add(obj); console.warn('prepare', p.name, e); }
    app.backfill.done++; showPrep();
    if (active && fresh >= 25) { fresh = 0; rebuildIndex(false); setStatus(); }   // people become recognisable as they are prepared
  }
  if (active && gen === app.bfGen) { rebuildIndex(false); setStatus(); }
}
async function backfill() {
  const gen = ++app.bfGen; if (!app.rec) return;
  while (app.bfBusy) await sleep(50);   // an earlier run stops at its next step
  if (gen !== app.bfGen) return;
  app.bfBusy = true;
  try {
    await fillModel(app.recModel, app.rec, gen, true);
    const other = app.recModel === 'accurate' ? 'fast' : 'accurate';
    if (gen === app.bfGen && missingFor(MODELS[other].key).length && await modelCached(other)) {
      const rec = await makeRecognizer(joinBuffers(await Promise.all(MODELS[other].parts.map((a) => getAsset(a, () => {})))));
      try { await fillModel(other, rec, gen, false); } finally { rec.close(); }
    }
  } catch (e) { console.warn('prepare', e); }
  app.bfBusy = false;
  if (gen === app.bfGen) { app.backfill = null; showPrep(); if (openSheetEl === $('#sheet-people') && app.pTab === 'people') renderPeopleList(); }
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
  save('rtpi-settings', s); renderCalib();
}
function bindSettings() {
  const s = app.settings;
  const sync = syncSettings;
  $('#s-margin').value = s.margin; $('#s-det').value = s.det; $('#s-max').value = s.max; $('#s-scores').checked = s.scores;
  $('#s-maybe').checked = s.maybe; $('#s-learn').checked = s.learn;
  $('#s-model').onchange = (e) => switchModel(e.target.value);
  $('#s-thr').oninput = (e) => { s['thr_' + (app.recModel || s.model)] = +e.target.value; sync(); };
  $('#s-margin').oninput = (e) => { s.margin = +e.target.value; sync(); };
  $('#s-det').onchange = (e) => { s.det = +e.target.value; sync(); };
  $('#s-max').onchange = (e) => { s.max = +e.target.value; sync(); };
  $('#s-scores').onchange = (e) => { s.scores = e.target.checked; sync(); };
  $('#s-maybe').onchange = (e) => { s.maybe = e.target.checked; sync(); app.tracks.forEach((t) => { if (t.emb) t.result = match(t.emb, t.deny); }); };
  $('#s-learn').onchange = (e) => { s.learn = e.target.checked; sync(); };
  $('#calib-apply').onclick = () => { const c = calib(app.recModel || s.model); if (c.thr == null) return; s['thr_' + (app.recModel || s.model)] = c.thr; sync(); toast(`Threshold set to ${c.thr.toFixed(2)}`); };
  armButton($('#calib-reset'), 'Tap again to forget the answers', () => { app.fb[app.recModel || s.model] = []; save('rtpi-fb', app.fb); renderCalib(); });
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
  try {
    app.hist = ((await DB.histAll()) || []).sort((a, b) => a.t - b.t);
    if (app.hist.length > 5000) { const old = app.hist.splice(0, app.hist.length - 5000); DB.histDel(old.map((h) => h.id)).catch(() => {}); }
  } catch { app.hist = []; }
  showNotice('Loading face models', `About ${app.settings.model === 'accurate' ? 57 : 27} MB, downloaded once and then kept on this phone.`, [], true);
  try { await loadModels(); }
  catch (e) { console.error(e); app.loadError = true; showNotice("Couldn't load the face models", e.message || String(e), [{ label: 'Reload', primary: true, fn: () => location.reload() }]); return; }
  await reloadPeople();
  window.__rtpi = app;
  if (new URLSearchParams(location.search).has('shared')) { hideNotice(); await takeShared(); }
  showNotice('Starting camera', 'Allow camera access when your browser asks.');
  if (await startCamera()) { updateNotice(); started = true; loop(); }
  syncSettings(); backfill();
  window.__rtpi = app; // for testing
})();
