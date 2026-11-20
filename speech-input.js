"use strict";

// Voice question, fully on the device: a small voice-activity model (Silero VAD)
// detects a short spoken phrase such as "ماهذا؟" and takes the picture. No audio
// leaves the device. app.js uses createSpeechInput({ canListen, onQuestion, onError }).

const VAD_SAMPLE_RATE = 16000;
const VAD_FRAME = 512;                          // 32 ms per frame
const FRAME_SECONDS = VAD_FRAME / VAD_SAMPLE_RATE;

// A question = a short phrase with a quiet moment before and after it.
// Long talking (a conversation, the TV) is ignored.
const SPEECH_ON = 0.5;                          // probability that starts a phrase
const SPEECH_OFF = 0.35;                        // probability below which it is quiet again
const MIN_PHRASE = 0.25;                        // seconds
const MAX_PHRASE = 2.0;
const QUIET_BEFORE = 0.7;
const QUIET_AFTER = 0.6;

const runtimeFolder = new URL("vendor/expanded-lite/", document.baseURI);
const runtimeCacheName = "misk-expanded-model-v5";   // shared with app.js, so nothing is downloaded twice
const vadModelUrl = new URL("vendor/vad/silero_vad.onnx", document.baseURI);

const debugSpeech = new URLSearchParams(location.search).has("debug");
function log(...parts) {
  if (debugSpeech) console.info("[voice]", ...parts);
}

const TEXT = {
  preparing: (percent) => `جاري تجهيز المايك… ${percent}٪`,
  listening: "أسمعك… اسأل: ماهذا؟",
  paused: "المايك متوقف مؤقتًا",
  unsupported: "متصفحك ما يدعم السؤال بالصوت. استخدم زر «ماهذا؟».",
  denied: "اسمح بالمايك من المتصفح أو استخدم زر «ماهذا؟».",
  failed: "تعذّر تجهيز المايك. تأكد من الاتصال وجرّب مرة ثانية، أو استخدم زر «ماهذا؟».",
};

// Loads the ONNX Runtime files (shared with the picture model) and the voice model.
async function loadVoiceModel(onProgress) {
  const cache = await window.caches?.open(runtimeCacheName).catch(() => null);
  async function fetchBytes(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed: ${url}`);
    return new Uint8Array(await response.arrayBuffer());
  }
  async function cachedPart(name, bytes) {
    const url = new URL(name, runtimeFolder).href;
    const cached = await cache?.match(url).catch(() => null);
    const data = cached ? new Uint8Array(await cached.arrayBuffer()) : null;
    if (data && data.byteLength === bytes) return data;
    const fresh = await fetchBytes(url);
    if (fresh.byteLength !== bytes) throw new Error("Incomplete runtime part");
    await cache?.put(url, new Response(fresh)).catch(() => {});
    return fresh;
  }
  async function assemble(entry, step) {
    const buffer = new Uint8Array(entry.bytes);
    let offset = 0;
    for (const { name, bytes } of entry.parts) {
      buffer.set(await cachedPart(name, bytes), offset);
      offset += bytes;
      step(bytes);
    }
    return buffer;
  }
  async function gunzip(bytes, expected) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    const result = new Uint8Array(await new Response(stream).arrayBuffer());
    if (result.byteLength !== expected) throw new Error("Incomplete runtime");
    return result;
  }

  const assets = JSON.parse(new TextDecoder().decode(await fetchBytes(new URL("assets.json", runtimeFolder))));
  const total = assets.runtime.bytes + assets.script.bytes + assets.module.bytes;
  let done = 0;
  const step = (bytes) => {
    done += bytes;
    onProgress(Math.min(99, Math.floor(100 * done / total)));
  };
  const runtime = await assemble(assets.runtime, step);
  const runtimeScript = await assemble(assets.script, step);
  const runtimeModule = await assemble(assets.module, step);
  const wasm = await gunzip(runtime, assets.runtime.expandedBytes);

  let model = await cache?.match(vadModelUrl.href).then((r) => r?.arrayBuffer()).catch(() => null);
  if (!model) {
    model = (await fetchBytes(vadModelUrl)).buffer;
    await cache?.put(vadModelUrl.href, new Response(model.slice(0))).catch(() => {});
  }
  return { runtimeScript, runtimeModule, wasm: wasm.buffer, model };
}

// One worker for the whole visit; preparing the model again is slow.
let vadWorker;
let vadReady;
function prepareVoiceModel(onProgress) {
  vadReady ??= (async () => {
    const files = await loadVoiceModel(onProgress);
    vadWorker = new Worker("vad-worker.js?v=1");
    await new Promise((resolve, reject) => {
      vadWorker.onmessage = ({ data }) => (data.error ? reject(new Error(data.error)) : resolve());
      vadWorker.onerror = (event) => reject(new Error(event.message));
      vadWorker.postMessage({ type: "prepare", ...files }, [files.wasm, files.model]);
    });
    return vadWorker;
  })().catch((error) => {
    vadReady = undefined;
    vadWorker?.terminate();
    vadWorker = undefined;
    throw error;
  });
  return vadReady;
}

// Turns the microphone's audio (usually 44.1 or 48 kHz) into 16 kHz samples.
function createResampler(inputRate) {
  const ratio = inputRate / VAD_SAMPLE_RATE;
  let position = 0;
  let previous = 0;
  return (input) => {
    const output = [];
    while (position < input.length) {
      const index = Math.floor(position);
      const fraction = position - index;
      const a = index === 0 ? previous : input[index - 1];
      const b = input[index];
      output.push(a + (b - a) * fraction);
      position += ratio;
    }
    position -= input.length;
    previous = input[input.length - 1];
    return output;
  };
}

// Mic bars driven by the real microphone level.
function createLevelBars() {
  const bars = [...(document.getElementById("mic-wave")?.children ?? [])];
  const weights = [0.35, 0.62, 0.83, 1, 0.83, 0.62, 0.35];
  let level = 0;
  return {
    show(rms, speaking) {
      const target = Math.min(1, Math.sqrt(Math.max(0, rms - 0.01) * 16)) * (speaking ? 1 : 0.6);
      level += (target - level) * (target > level ? 0.7 : 0.25);
      for (const [i, bar] of bars.entries()) bar.style.transform = `scaleY(${0.16 + level * 0.84 * weights[i]})`;
    },
    stop() {
      level = 0;
      for (const bar of bars) bar.style.transform = "scaleY(0.16)";
    },
  };
}

// Decides when a phrase counts as a question (see the constants at the top).
function createPhraseDetector(onPhrase) {
  let speaking = false;
  let quietFrames = 0;
  let phraseFrames = 0;
  let quietBefore = false;
  return {
    reset() {
      speaking = false;
      quietFrames = 0;      // after a pause, require a fresh quiet moment before the next phrase
      phraseFrames = 0;
    },
    push(probability) {
      if (!speaking) {
        if (probability >= SPEECH_ON) {
          speaking = true;
          quietBefore = quietFrames * FRAME_SECONDS >= QUIET_BEFORE;
          phraseFrames = 1;
          quietFrames = 0;
        } else {
          quietFrames += 1;
        }
        return speaking;
      }
      if (probability >= SPEECH_OFF) {
        phraseFrames += 1 + quietFrames;
        quietFrames = 0;
        return true;
      }
      quietFrames += 1;
      if (quietFrames * FRAME_SECONDS >= QUIET_AFTER) {
        const seconds = phraseFrames * FRAME_SECONDS;
        speaking = false;
        log(`phrase ${seconds.toFixed(2)} s, quiet before: ${quietBefore}`);
        if (quietBefore && seconds >= MIN_PHRASE && seconds <= MAX_PHRASE) onPhrase(seconds);
      }
      return speaking;
    },
  };
}

function createSpeechInput({ canListen, onQuestion, onError }) {
  const state = document.getElementById("mic-state");
  const AudioEngine = window.AudioContext ?? window.webkitAudioContext;
  const bars = createLevelBars();
  let token = 0;
  let audio;               // { context, stream, source, processor, sink }
  let paused = true;
  let speaking = false;
  let speechUnlocked = false;
  let pending = [];

  const detector = createPhraseDetector((seconds) => {
    log(`question detected (${seconds.toFixed(2)} s)`);
    pause();
    onQuestion();
  });

  function setText(text) {
    if (state) state.textContent = text;
  }

  // iPhones only speak after a tap started speech once.
  function unlockSpeech() {
    if (speechUnlocked || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return;
    speechUnlocked = true;
    try {
      const silent = new SpeechSynthesisUtterance(" ");
      silent.volume = 0;
      window.speechSynthesis.speak(silent);
    } catch {}
  }

  function pause() {
    paused = true;
    pending = [];
    speaking = false;
    detector.reset();
    vadWorker?.postMessage({ type: "reset" });
    bars.stop();
  }

  function disable() {
    token += 1;
    pause();
    if (!audio) return;
    const { context, stream, source, processor, sink } = audio;
    audio = undefined;
    processor.onaudioprocess = null;
    try { source.disconnect(); processor.disconnect(); sink.disconnect(); } catch {}
    stream.getTracks().forEach((track) => track.stop());
    context.close().catch(() => {});
  }

  function refresh() {
    if (!audio) return;
    if (!canListen()) {
      if (!paused) pause();
      setText(TEXT.paused);
    } else if (paused) {
      paused = false;
      setText(TEXT.listening);
    }
  }

  function handleAudio(event, resample) {
    const input = event.inputBuffer.getChannelData(0);
    let sum = 0;
    for (const value of input) sum += value * value;
    const rms = Math.sqrt(sum / input.length);
    if (paused) return;
    bars.show(rms, speaking);
    for (const sample of resample(input)) pending.push(sample);
    const usable = pending.length - (pending.length % VAD_FRAME);
    if (usable && vadWorker) {
      const samples = Float32Array.from(pending.slice(0, usable));
      pending = pending.slice(usable);
      vadWorker.postMessage({ type: "frames", samples }, [samples.buffer]);
    }
  }

  async function enable() {
    disable();
    const myToken = token;
    unlockSpeech();
    if (!AudioEngine || !navigator.mediaDevices?.getUserMedia || !window.Worker
        || !window.WebAssembly || !window.DecompressionStream) {
      onError(TEXT.unsupported);
      return;
    }
    // Created during the tap that turned the mic on, so phones allow audio.
    const context = new AudioEngine();
    const resumed = context.resume().catch(() => {});
    setText(TEXT.preparing(0));
    let stream;
    try {
      [stream] = await Promise.all([
        navigator.mediaDevices.getUserMedia({
          video: false,
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        }),
        prepareVoiceModel((percent) => { if (myToken === token) setText(TEXT.preparing(percent)); }),
      ]);
      await resumed;
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      context.close().catch(() => {});
      if (myToken !== token) return;
      log("could not start:", error);
      onError(error?.name === "NotAllowedError" ? TEXT.denied : TEXT.failed);
      return;
    }
    if (myToken !== token) {
      stream.getTracks().forEach((track) => track.stop());
      context.close().catch(() => {});
      return;
    }

    vadWorker.onmessage = ({ data }) => {
      if (myToken !== token || paused) return;
      if (data.error) {
        log("voice model error:", data.error);
        return;
      }
      for (const probability of data.probs) speaking = detector.push(probability);
    };

    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const sink = context.createGain();
    sink.gain.value = 0;      // keeps the processor running without playing the mic back
    const resample = createResampler(context.sampleRate);
    processor.onaudioprocess = (event) => handleAudio(event, resample);
    source.connect(processor);
    processor.connect(sink);
    sink.connect(context.destination);
    audio = { context, stream, source, processor, sink };
    stream.getAudioTracks()[0]?.addEventListener("ended", () => {
      if (myToken === token) onError(TEXT.failed);
    }, { once: true });
    log(`listening on the device (mic ${context.sampleRate} Hz → 16 kHz)`);
    refresh();
  }

  return { enable, disable, refresh };
}
