"use strict";

// Runs the voice models on the device. Silero VAD gives, for each 32 ms frame of
// 16 kHz audio, the probability that it is speech. Moonshine Tiny (Arabic and
// English) turns a short phrase into text, so the app can check that it really
// is the question. No audio leaves the device.

const FRAME = 512;
const CONTEXT = 64;
const SAMPLE_RATE = 16000;
const KEEP_SECONDS = 8;                 // recent audio kept for transcription
const MAX_TOKENS = 24;
const LAYERS = 6;
const HEADS = 8;
const HEAD_SIZE = 36;
const START_TOKEN = 1;
const END_TOKEN = 2;

let session;
let state;
let context;
let readers = [];                      // [{ lang, encoder, decoder, vocab }]
let history = new Float32Array(KEEP_SECONDS * SAMPLE_RATE);
let received = 0;                      // samples received since the start

function reset() {
  state = new ort.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
  context = new Float32Array(CONTEXT);
}

async function prepare({ runtimeScript, runtimeModule, wasm, model, speech }) {
  const scriptUrl = URL.createObjectURL(new Blob([runtimeScript], { type: "text/javascript" }));
  const moduleUrl = URL.createObjectURL(new Blob([runtimeModule], { type: "text/javascript" }));
  try {
    importScripts(scriptUrl);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: moduleUrl };
    ort.env.wasm.wasmBinary = wasm;
    const options = { executionProviders: ["wasm"] };
    session = await ort.InferenceSession.create(new Uint8Array(model), options);
    for (const { lang, encoder, decoder, vocab } of speech) {
      readers.push({
        lang,
        encoder: await ort.InferenceSession.create(new Uint8Array(encoder), options),
        decoder: await ort.InferenceSession.create(new Uint8Array(decoder), options),
        vocab: JSON.parse(new TextDecoder().decode(vocab)),
      });
    }
    ort.env.wasm.wasmBinary = undefined;
  } finally {
    URL.revokeObjectURL(scriptUrl);
    URL.revokeObjectURL(moduleUrl);
  }
  reset();
}

const sampleRate = () => new ort.Tensor("int64", BigInt64Array.from([16000n]), []);

function remember(samples) {
  const size = history.length;
  for (let i = 0; i < samples.length; i++) history[(received + i) % size] = samples[i];
  received += samples.length;
}

async function detect(samples) {
  const first = (received - samples.length) / FRAME;
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
  return { first, probs };
}

// Audio between two frame numbers, with 0.2 s extra on each side.
function phraseAudio(startFrame, endFrame) {
  const margin = 0.2 * SAMPLE_RATE;
  const from = Math.max(0, received - history.length, startFrame * FRAME - margin);
  const to = Math.min(received, endFrame * FRAME + margin);
  const out = new Float32Array(Math.max(0, to - from));
  for (let i = 0; i < out.length; i++) out[i] = history[(from + i) % history.length];
  return out;
}

function decodeText(ids, vocab) {
  const bytes = [];
  const encoder = new TextEncoder();
  for (const id of ids) {
    const piece = vocab[id] ?? "";
    const byte = /^<0x([0-9A-F]{2})>$/.exec(piece);
    if (byte) bytes.push(parseInt(byte[1], 16));
    else bytes.push(...encoder.encode(piece.replaceAll("▁", " ")));
  }
  return new TextDecoder().decode(new Uint8Array(bytes)).trim();
}

// Moonshine: encoder once, then greedy decoding with the key/value cache.
async function transcribe(reader, audio) {
  const { last_hidden_state: hidden } = await reader.encoder.run({
    input_values: new ort.Tensor("float32", audio, [1, audio.length]),
  });
  const past = {};
  const empty = new ort.Tensor("float32", new Float32Array(0), [1, HEADS, 0, HEAD_SIZE]);
  for (let l = 0; l < LAYERS; l++) {
    for (const part of ["decoder", "encoder"]) {
      past[`past_key_values.${l}.${part}.key`] = empty;
      past[`past_key_values.${l}.${part}.value`] = empty;
    }
  }
  const ids = [START_TOKEN];
  for (let step = 0; step < MAX_TOKENS; step++) {
    const outputs = await reader.decoder.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(ids.at(-1))]), [1, 1]),
      encoder_hidden_states: hidden,
      use_cache_branch: new ort.Tensor("bool", new Uint8Array([step > 0 ? 1 : 0]), [1]),
      ...past,
    });
    const logits = outputs.logits.data;
    const vocabSize = outputs.logits.dims[2];
    let best = 0;
    for (let i = 1; i < vocabSize; i++) if (logits[i] > logits[best]) best = i;
    for (const [name, value] of Object.entries(outputs)) {
      if (!name.startsWith("present.")) continue;
      if (step > 0 && name.includes(".encoder.")) continue;    // the encoder cache is made once
      past[name.replace("present.", "past_key_values.")] = value;
    }
    if (best === END_TOKEN) break;
    ids.push(best);
  }
  return decodeText(ids.slice(1), reader.vocab);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "prepare") {
      await prepare(data);
      self.postMessage({ ready: true });
    } else if (data.type === "reset") {
      if (session) reset();
    } else if (data.type === "frames") {
      remember(data.samples);
      self.postMessage(await detect(data.samples));
    } else if (data.type === "transcribe") {
      const audio = phraseAudio(data.start, data.end);
      const started = performance.now();
      const texts = [];
      for (const reader of readers) texts.push({ lang: reader.lang, text: await transcribe(reader, audio) });
      self.postMessage({ texts, id: data.id, ms: Math.round(performance.now() - started) });
    }
  } catch (error) {
    self.postMessage({ error: error?.message || String(error), id: data.id });
  }
};
