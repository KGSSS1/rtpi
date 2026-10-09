// Offline cache: after the first visit the app, models and runtime load from the phone.
const CACHE = 'rtpi-v3';
const FILES = ['./', 'index.html', 'app.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-180.png',
  'vendor/ort.wasm.min.js', 'vendor/ort-wasm-simd-threaded.mjs'];
// the large model and runtime files are downloaded and kept by the page itself (cache 'rtpi-assets-…')
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES.map((f) => new Request(f, { cache: 'reload' })))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE && k !== 'rtpi-share' && !k.startsWith('rtpi-assets')).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
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
  if (/\.(onnx|wasm)$/.test(url.pathname)) return; // handled by the page
  if (url.pathname.endsWith('/') || url.pathname.endsWith('/index.html') || url.pathname.endsWith('app.js')) {
    // app shell: use the network when online so updates arrive straight away, saved copy when offline
    e.respondWith(fetch(e.request).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); } return res; })
      .catch(() => caches.match(e.request, { ignoreSearch: true })));
    return;
  }
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  })));
});
