/* RTPI — real-time person identification, fully on-device.
   Pipeline per frame: SCRFD-500M face detection -> 5-point similarity alignment (ArcFace 112x112)
   -> MobileFaceNet (w600k_mbf) 512-d embedding -> cosine match against the local people list. */
'use strict';

const $ = (s) => document.querySelector(s);
const ARC = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
const DIM = 512;
const DEFAULTS = { thr: 0.42, margin: 0.05, det: 384, max: 3, scores: false, facing: 'user' };

const app = {
  det: null, rec: null,
  people: [], E: new Float32Array(0),
  settings: load('rtpi-settings', DEFAULTS),
  stream: null, running: false,
  tracks: [], nextTrackId: 1, hits: [],
  fps: 0, frameW: 0, frameH: 0,
  last: null,              // last processed frame canvas snapshot info for "add from camera"
  addDraft: null,
};

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

async function reloadPeople() {
  let rows = [];
  try { rows = (await DB.all()) || []; } catch (e) { console.warn('storage unavailable', e); rows = app.people; }
  rows.sort((a, b) => a.name.localeCompare(b.name));
  app.people = rows;
  const E = new Float32Array(rows.length * DIM);
  rows.forEach((p, i) => E.set(p.emb instanceof Float32Array ? p.emb : new Float32Array(p.emb), i * DIM));
  app.E = E;
  app.tracks.forEach((t) => (t.label = null));
  $('#count-badge').textContent = rows.length; $('#count-badge').hidden = rows.length === 0;
  renderPeopleList();
  updateNotice();
}

/* ---------- models ---------- */
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total) { const b = await res.arrayBuffer(); onBytes(b.byteLength); return b; }
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onBytes(value.length); }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
}

async function loadModels() {
  ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
  ort.env.logLevel = 'error';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const sizes = { det: 2524817, rec: 13616099 }; const total = sizes.det + sizes.rec; let got = 0;
  const bar = $('#notice-progress > div');
  const tick = (n) => { got += n; bar.style.width = Math.min(100, (got / total) * 100).toFixed(1) + '%'; };
  const [detBuf, recBuf] = await Promise.all([fetchWithProgress('models/det_500m.onnx', tick), fetchWithProgress('models/w600k_mbf.onnx', tick)]);
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 };
  app.det = await ort.InferenceSession.create(detBuf, opts);
  app.rec = await ort.InferenceSession.create(recBuf, opts);
}

/* ---------- detection (SCRFD) ---------- */
const detCanvas = canvas(640, 640), detCtx = detCanvas.getContext('2d', { willReadFrequently: true });

async function detect(src, W, H, S, thresh = 0.5) {
  if (detCanvas.width !== S) { detCanvas.width = detCanvas.height = S; }
  const s = S / Math.max(W, H);
  detCtx.fillStyle = '#000'; detCtx.fillRect(0, 0, S, S);
  detCtx.drawImage(src, 0, 0, W, H, 0, 0, Math.round(W * s), Math.round(H * s));
  const px = detCtx.getImageData(0, 0, S, S).data; const n = S * S; const x = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { x[i] = (px[j] - 127.5) / 128; x[n + i] = (px[j + 1] - 127.5) / 128; x[2 * n + i] = (px[j + 2] - 127.5) / 128; }
  const out = await app.det.run({ [app.det.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, S, S]) });
  const o = app.det.outputNames.map((k) => out[k].data);
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

async function embedMany(src, faces) {
  if (!faces.length) return [];
  const n = 112 * 112, x = new Float32Array(faces.length * 3 * n);
  faces.forEach((f, k) => {
    const px = alignedPixels(src, f.pts), o = k * 3 * n;
    for (let i = 0, j = 0; i < n; i++, j += 4) { x[o + i] = (px[j] - 127.5) / 127.5; x[o + n + i] = (px[j + 1] - 127.5) / 127.5; x[o + 2 * n + i] = (px[j + 2] - 127.5) / 127.5; }
  });
  const out = await app.rec.run({ [app.rec.inputNames[0]]: new ort.Tensor('float32', x, [faces.length, 3, 112, 112]) });
  const d = out[app.rec.outputNames[0]].data;
  return faces.map((_, k) => normalize(Float32Array.from(d.subarray(k * DIM, (k + 1) * DIM))));
}

function match(e) {
  const N = app.people.length; let best = -1, second = -1, bi = -1;
  for (let p = 0; p < N; p++) {
    let s = 0; const o = p * DIM; for (let i = 0; i < DIM; i++) s += app.E[o + i] * e[i];
    if (s > best) { second = best; best = s; bi = p; } else if (s > second) second = s;
  }
  const ok = bi >= 0 && best >= app.settings.thr && best - Math.max(second, 0) >= app.settings.margin;
  return { person: ok ? app.people[bi] : null, closest: bi >= 0 ? app.people[bi] : null, score: best, second };
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
    if (!app.stream || video.readyState < 2 || !video.videoWidth || document.hidden) { await sleep(120); continue; }
    const vw = video.videoWidth, vh = video.videoHeight, k = Math.min(1, MAX_SIDE / Math.max(vw, vh));
    const W = Math.round(vw * k), H = Math.round(vh * k);
    if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
    frameCtx.drawImage(video, 0, 0, W, H);
    app.frameW = W; app.frameH = H; app.frameScale = k;
    try {
      const faces = (await detect(frameCanvas, W, H, +app.settings.det)).filter((f) => f.box[2] - f.box[0] > 24);
      faces.sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]));
      const use = faces.slice(0, +app.settings.max);
      const embs = await embedMany(frameCanvas, use);
      updateTracks(use, embs);
    } catch (e) { console.error(e); }
    draw();
    const now = performance.now(); app.fps = app.fps * 0.8 + (1000 / Math.max(1, now - lastT)) * 0.2; lastT = now;
    setStatus();
    await new Promise((r) => requestAnimationFrame(r));
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function updateTracks(faces, embs) {
  const used = new Set(); const next = [];
  faces.forEach((f, i) => {
    let bestT = null, bestI = 0.25;
    for (const t of app.tracks) { if (used.has(t)) continue; const v = iou(t.box, f.box); if (v > bestI) { bestI = v; bestT = t; } }
    const t = bestT || { id: app.nextTrackId++, emb: null, seen: 0 };
    used.add(t);
    // smooth the identity over frames: running mean of embeddings, re-normalised
    if (!t.emb) t.emb = Float32Array.from(embs[i]);
    else { const w = Math.min(t.seen, 6); for (let d = 0; d < DIM; d++) t.emb[d] = t.emb[d] * w + embs[i][d]; normalize(t.emb); }
    t.seen++; t.box = f.box; t.pts = f.pts; t.raw = embs[i]; t.missed = 0;
    t.result = match(t.emb);
    next.push(t);
  });
  // keep briefly-lost tracks for a couple of frames so labels don't flicker
  for (const t of app.tracks) if (!used.has(t) && ++t.missed <= 2) next.push(t);
  app.tracks = next;
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
    else { title = 'Unknown'; line2 = app.settings.scores && r.closest ? `closest: ${r.closest.name}` : 'Tap to add'; }
    if (app.settings.scores && r.closest) title += `  ${r.score.toFixed(2)}`;
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
  if (r.person) openPerson(r.person, r.score);
  else openAdd(snapshotTrack(h.track));
});

/* ---------- status + notice ---------- */
function setStatus() {
  const n = app.people.length, named = app.tracks.filter((t) => t.result?.person && !t.missed).length;
  $('#status').innerHTML = `<b>${n}</b> people · ${app.fps.toFixed(1)} fps${named ? ` · <b>${named}</b> named` : ''}`;
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
function openSheet(id) { closeSheet(); openSheetEl = $(id); openSheetEl.hidden = false; $('#scrim').hidden = false; }
function closeSheet() { if (openSheetEl) openSheetEl.hidden = true; openSheetEl = null; $('#scrim').hidden = true; }
$('#scrim').onclick = closeSheet;
addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });

function openPerson(p, score) {
  const thr = app.settings.thr;
  $('#person-body').innerHTML = `
    <div class="person">
      <img src="${esc(p.thumb)}" alt="">
      <div class="who">
        <div class="name">${esc(p.name)}${p.source === 'added' ? '<span class="tag">added</span>' : ''}</div>
        ${p.headline ? `<p class="head">${esc(p.headline)}</p>` : ''}
        <div class="meta">${[p.organisation, p.location].filter(Boolean).map(esc).join(' · ')}</div>
      </div>
    </div>
    ${score != null ? `<div class="match"><span>match</span><div class="meter"><div style="width:${Math.max(0, Math.min(1, score)) * 100}%"></div><i style="left:${thr * 100}%"></i></div><span>${score.toFixed(2)}</span></div>` : ''}
    ${p.link ? `<a class="link" href="${esc(p.link)}" target="_blank" rel="noopener">${esc(p.link.replace(/^https?:\/\/(www\.)?/, ''))}</a>` : ''}
    <button class="btn danger" id="btn-del-person">Remove from this phone</button>`;
  armButton($('#btn-del-person'), 'Tap again to remove', async () => { await DB.del(p.id); await reloadPeople(); closeSheet(); toast(`Removed ${p.name}`); });
  openSheet('#sheet-person');
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
    if (pack.format !== 'face-pack' || pack.model !== 'w600k_mbf' || !Array.isArray(pack.people)) throw new Error('This file is not an RTPI face pack.');
    const rows = pack.people.map((p) => {
      const emb = b64ToF32(p.emb); if (emb.length !== DIM) throw new Error(`Bad face data for ${p.name}.`);
      return { id: String(p.id), name: p.name, headline: p.headline || '', organisation: p.organisation || '', location: p.location || '', link: p.link || '', thumb: p.thumb || '', emb, source: p.source || 'pack' };
    });
    await DB.putMany(rows).catch(() => { app.people = rows; });
    await reloadPeople();
    toast(`Imported ${rows.length} people`);
  } catch (err) { toast(err.message || 'Could not read that file', 4000); }
};
$('#btn-export').onclick = () => {
  const pack = { format: 'face-pack', version: 1, model: 'w600k_mbf', dim: DIM, created: new Date().toISOString().slice(0, 16).replace('T', ' '),
    people: app.people.map((p) => ({ id: p.id, name: p.name, headline: p.headline, organisation: p.organisation, location: p.location, link: p.link, thumb: p.thumb, source: p.source, emb: f32ToB64(p.emb instanceof Float32Array ? p.emb : new Float32Array(p.emb)) })) };
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
  return { thumb: c.toDataURL('image/jpeg', 0.82), emb: Float32Array.from(t.raw) };
}
function openAdd(draft) {
  app.addDraft = draft || null;
  $('#add-form').reset();
  $('#add-preview').src = draft ? draft.thumb : '';
  $('#add-sub').textContent = draft ? 'Face captured. Add their details.' : 'Use the face in front of the camera now, or pick a photo.';
  openSheet('#sheet-add');
}
$('#btn-add').onclick = () => {
  const t = app.tracks.filter((t) => !t.missed).sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
  openAdd(t && !t.result?.person ? snapshotTrack(t) : null);
};
$('#add-from-camera').onclick = () => {
  const t = app.tracks.filter((t) => !t.missed).sort((a, b) => (b.box[2] - b.box[0]) - (a.box[2] - a.box[0]))[0];
  if (!t) { toast('No face in view. Point the camera at the person.'); return; }
  app.addDraft = snapshotTrack(t); $('#add-preview').src = app.addDraft.thumb; $('#add-sub').textContent = 'Face captured. Add their details.';
};
$('#add-from-photo').onclick = () => $('#file-photo').click();
$('#file-photo').onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const bmp = await createImageBitmap(f);
    const k = Math.min(1, 1600 / Math.max(bmp.width, bmp.height)), W = Math.round(bmp.width * k), H = Math.round(bmp.height * k);
    const c = canvas(W, H); c.getContext('2d').drawImage(bmp, 0, 0, W, H);
    const faces = await detect(c, W, H, 640);
    if (!faces.length) { toast('No face found in that photo. Try a clearer, front-facing one.', 3500); return; }
    const f0 = faces.sort((a, b) => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) - (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]))[0];
    const [emb] = await embedMany(c, [f0]);
    const [x1, y1, x2, y2] = f0.box, cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, half = Math.max(x2 - x1, y2 - y1) * 0.9;
    const tc = canvas(160, 160); tc.getContext('2d').drawImage(c, cx - half, cy - half, half * 2, half * 2, 0, 0, 160, 160);
    app.addDraft = { thumb: tc.toDataURL('image/jpeg', 0.82), emb };
    $('#add-preview').src = app.addDraft.thumb; $('#add-sub').textContent = faces.length > 1 ? 'Several faces found; used the largest.' : 'Face found. Add their details.';
  } catch (err) { console.error(err); toast('Could not read that photo.'); }
};
$('#add-form').onsubmit = async (e) => {
  e.preventDefault();
  if (!app.addDraft) { toast('Capture a face or pick a photo first.'); return; }
  const name = $('#add-name').value.trim(); if (!name) return;
  const dup = app.people.length ? match(app.addDraft.emb) : null;
  const p = { id: 'u' + Date.now().toString(36), name, headline: $('#add-headline').value.trim(), organisation: $('#add-org').value.trim(),
    location: $('#add-location').value.trim(), link: $('#add-link').value.trim(), thumb: app.addDraft.thumb, emb: app.addDraft.emb, source: 'added' };
  await DB.putMany([p]).catch(() => app.people.push(p));
  await reloadPeople(); closeSheet();
  toast(dup?.person ? `Saved ${name}. Looks similar to ${dup.person.name}; check it's not a duplicate.` : `Saved ${name}`, dup?.person ? 4500 : 2200);
};

/* settings */
function bindSettings() {
  const s = app.settings;
  const sync = () => { $('#v-thr').textContent = (+s.thr).toFixed(2); $('#v-margin').textContent = (+s.margin).toFixed(2); save('rtpi-settings', s); };
  $('#s-thr').value = s.thr; $('#s-margin').value = s.margin; $('#s-det').value = s.det; $('#s-max').value = s.max; $('#s-scores').checked = s.scores;
  $('#s-thr').oninput = (e) => { s.thr = +e.target.value; sync(); };
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
  if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
  try { await DB.open(); } catch (e) { console.warn('IndexedDB unavailable; people will not be kept between visits', e); }
  showNotice('Loading face models', 'About 16 MB, downloaded once and then kept on this phone.', [], true);
  try { await loadModels(); }
  catch (e) { console.error(e); showNotice('Models failed to load', 'Check your connection and reload. After the first load the app works offline.', [{ label: 'Reload', primary: true, fn: () => location.reload() }]); return; }
  await reloadPeople();
  showNotice('Starting camera', 'Allow camera access when your browser asks.');
  if (await startCamera()) { updateNotice(); loop(); }
  window.__rtpi = app; // for testing
})();
