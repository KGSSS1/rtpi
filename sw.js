// Offline cache: after the first visit the app, models and runtime load from the phone.
const CACHE = 'rtpi-v2';
const FILES = ['./', 'index.html', 'app.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-180.png',
  'vendor/ort.wasm.min.js', 'vendor/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.wasm',
  'models/det_500m.onnx', 'models/w600k_mbf.onnx'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES.map((f) => new Request(f, { cache: 'reload' })))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE && k !== 'rtpi-share').map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // a photo shared to RTPI from another app (Android share menu)
  if (e.request.method === 'POST' && url.pathname.endsWith('/share-target')) {
    e.respondWith((async () => {
      try {
        const form = await e.request.formData(); const file = form.get('image');
        if (file) await (await caches.open('rtpi-share')).put('shared-image', new Response(file, { headers: { 'content-type': file.type || 'image/jpeg' } }));
      } catch (err) { /* fall through to the app */ }
      return Response.redirect(new URL('./?shared=1', self.registration.scope).href, 303);
    })());
    return;
  }
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  })));
});
