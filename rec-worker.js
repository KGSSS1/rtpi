/* Face recognition runs here, off the main thread, so the camera view never freezes. */
importScripts('vendor/ort.wasm.min.js');
let session = null;
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      ort.env.wasm.wasmPaths = m.base + 'vendor/';
      ort.env.wasm.wasmBinary = m.wasm;
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';
      session = await ort.InferenceSession.create(m.model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 });
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'embed') {
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', m.x, [m.n, 3, 112, 112]) });
      const data = Float32Array.from(out[session.outputNames[0]].data);
      self.postMessage({ type: 'emb', id: m.id, data }, [data.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
