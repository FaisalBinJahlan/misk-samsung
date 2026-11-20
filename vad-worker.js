"use strict";

// Runs the Silero voice-activity model on the device. Receives 16 kHz audio in
// 512-sample frames and returns, for each frame, the probability that it is speech.

const FRAME = 512;
const CONTEXT = 64;
let session;
let state;
let context;

function reset() {
  state = new ort.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
  context = new Float32Array(CONTEXT);
}

async function prepare({ runtimeScript, runtimeModule, wasm, model }) {
  const scriptUrl = URL.createObjectURL(new Blob([runtimeScript], { type: "text/javascript" }));
  const moduleUrl = URL.createObjectURL(new Blob([runtimeModule], { type: "text/javascript" }));
  try {
    importScripts(scriptUrl);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: moduleUrl };
    ort.env.wasm.wasmBinary = wasm;
    session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ["wasm"] });
    ort.env.wasm.wasmBinary = undefined;
  } finally {
    URL.revokeObjectURL(scriptUrl);
    URL.revokeObjectURL(moduleUrl);
  }
  reset();
}

const sampleRate = () => new ort.Tensor("int64", BigInt64Array.from([16000n]), []);

async function detect(samples) {
  const probs = [];
  for (let i = 0; i + FRAME <= samples.length; i += FRAME) {
    const input = new Float32Array(CONTEXT + FRAME);
    input.set(context, 0);
    input.set(samples.subarray(i, i + FRAME), CONTEXT);
    const result = await session.run({
      input: new ort.Tensor("float32", input, [1, CONTEXT + FRAME]),
      state,
      sr: sampleRate(),
    });
    probs.push(result.output.data[0]);
    state = result.stateN;
    context = input.slice(input.length - CONTEXT);
  }
  return probs;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "prepare") {
      await prepare(data);
      self.postMessage({ ready: true });
    } else if (data.type === "reset") {
      if (session) reset();
    } else if (data.type === "frames") {
      self.postMessage({ probs: await detect(data.samples) });
    }
  } catch (error) {
    self.postMessage({ error: error?.message || String(error) });
  }
};
