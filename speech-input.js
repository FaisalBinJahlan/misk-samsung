"use strict";

// Voice question, fully on the device. Silero VAD finds short spoken phrases;
// Moonshine Tiny (Arabic and English) reads each phrase, and only the question
// ("ماهذا؟" in any dialect, or "what is this?") takes the picture. No audio leaves
// the device. app.js uses createSpeechInput({ canListen, onQuestion, onError }).

const VAD_SAMPLE_RATE = 16000;
const VAD_FRAME = 512;                          // 32 ms per frame
const FRAME_SECONDS = VAD_FRAME / VAD_SAMPLE_RATE;

// A phrase = speech followed by a quiet moment. Phrases of this length are read
// by the speech model; longer talk (a conversation, the TV) is ignored.
const SPEECH_ON = 0.5;                          // probability that starts a phrase
const SPEECH_OFF = 0.35;                        // probability below which it is quiet again
const MIN_PHRASE = 0.25;                        // seconds
const MAX_PHRASE = 3.0;
const QUIET_AFTER = 0.6;

const runtimeFolder = new URL("vendor/expanded-lite/", document.baseURI);
const runtimeCacheName = "misk-expanded-model-v5";   // shared with app.js, so nothing is downloaded twice
const vadModelUrl = new URL("vendor/vad/silero_vad.onnx", document.baseURI);
const speechFolder = new URL("vendor/moonshine/", document.baseURI);
const SPEECH_LANGS = ["ar", "en"];

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

// Lowercases, removes diacritics and unifies letter forms (أ/إ/آ → ا, ى → ي, ة → ه).
function normalizeArabic(text) {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u02BC]/g, "'")
    .replace(/[ً-ٰٟـ]/g, "")
    .replace(/[أإآٱ]/g, "ا")
    .replace(/ى/g, "ي")
    .replace(/ة/g, "ه")
    .replace(/[^\p{L}\p{N}\s']/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// A question = question word + "this" word, in either order, or a question about the name.
const QUESTION_WORDS_DIALECT = ["وش", "وشو", "ويش", "ايش", "اش", "اشنو", "شو", "شنو", "شنهو", "شني", "شنهي", "شنوه", "شنوا", "شن", "واش"];
const QUESTION_WORDS = ["ما", "ماذا", ...QUESTION_WORDS_DIALECT];
const THIS_WORDS = ["هذا", "هذه", "هذي", "هاذا", "هاذه", "هاذي", "هاذ", "هاد", "هادا", "هادي", "هاي", "هيدا", "هيدي", "هدا", "هدي", "ذاك", "هذاك", "ذيك", "هذيك"];
// Short forms only count after a dialect question word, so "ماذا تفعل" is not a question.
const THIS_WORDS_SHORT = ["ذا", "ذي", "ده", "دا", "دي"];
const NAME_WORDS = ["اسمه", "اسمها", "اسمو"];
const any = (words) => `(?:${words.join("|")})`;
const START = "(?:^| )";
const END = "(?: |$)";
// Allowed in the middle: "وش يكون هذا". Only these words, so "ما عندي هذا" is not a question.
const FILLER_WORDS = ["هو", "هي", "يكون", "تكون", "يعني", "بالضبط", "طيب"];
const HE_SHE = `(?:${any(FILLER_WORDS)} ){0,2}`;
const QUESTION_PATTERNS = [
  new RegExp(`${START}${any(QUESTION_WORDS)} ?${HE_SHE}${any(THIS_WORDS)}${END}`),            // ما هذا، شو هاد
  new RegExp(`${START}${any(QUESTION_WORDS_DIALECT)} ?${HE_SHE}${any(THIS_WORDS_SHORT)}${END}`), // وش ذا، شنو دا
  new RegExp(`${START}ايه ?(?:ده|دي|دا)${END}`),                                                // ايه ده ("ايه" alone = yes)
  new RegExp(`${START}${any([...THIS_WORDS, ...THIS_WORDS_SHORT])} ${any(QUESTION_WORDS_DIALECT)}${END}`), // هذا وش
  new RegExp(`${START}(?:ده|دي|دا) ايه${END}`),                                                 // ده ايه
  new RegExp(`${START}${any([...QUESTION_WORDS, "ايه"])} ?اسم ${any([...THIS_WORDS, ...THIS_WORDS_SHORT])}${END}`), // وش اسم هذا
  new RegExp(`${START}${any(QUESTION_WORDS)} ?${any(NAME_WORDS)}${END}`),                       // شو اسمو (not اسمك)
  /(?:^| )(?:what|wat|wut)(?:'s|s)?(?: exactly| actually)?(?: is| was|'s)?(?: the name of)? (?:this|that|it|dis|dat)(?: |$)/, // what is this
  /(?:^| )what(?:'s| is) (?:this|that|it) (?:called|named)(?: |$)/,                             // what is this called
  /(?:^| )(?:what are|what're) (?:these|those|they)(?: |$)/,                                     // what are these
  /(?:^| )what do you call (?:this|that|it)(?: |$)/,                                             // what do you call this
  /(?:^| )(?:وات|واط|وت|واتس|وتس|واتز|واتيز|وتيز) ?(?:از|ايز|إز)? ?(?:ذس|ذيس|ذيز|ذز|ديس|زيس|ذات|دات)(?: |$)/, // English heard as "وات از ذس"
];

function isQuestion(transcript) {
  const text = normalizeArabic(transcript);
  return QUESTION_PATTERNS.some((pattern) => pattern.test(text));
}

// Loads the ONNX Runtime files (shared with the picture model) and the voice models.
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

  // Voice model files, kept in the same cache.
  async function cachedFile(url, bytes, step) {
    const cached = await cache?.match(url.href).catch(() => null);
    const data = cached ? new Uint8Array(await cached.arrayBuffer()) : null;
    if (data && data.byteLength === bytes) {
      step(bytes);
      return data.buffer;
    }
    const fresh = await fetchBytes(url);
    if (fresh.byteLength !== bytes) throw new Error(`Incomplete file: ${url}`);
    await cache?.put(url.href, new Response(fresh)).catch(() => {});
    step(bytes);
    return fresh.buffer;
  }

  const assets = JSON.parse(new TextDecoder().decode(await fetchBytes(new URL("assets.json", runtimeFolder))));
  const speechAssets = JSON.parse(new TextDecoder().decode(await fetchBytes(new URL("assets.json", speechFolder))));
  const speechBytes = SPEECH_LANGS.reduce((sum, lang) => sum + Object.values(speechAssets[lang]).reduce((a, b) => a + b, 0), 0);
  const total = assets.runtime.bytes + assets.script.bytes + assets.module.bytes + speechAssets.silero + speechBytes;
  let done = 0;
  const step = (bytes) => {
    done += bytes;
    onProgress(Math.min(99, Math.floor(100 * done / total)));
  };
  const runtime = await assemble(assets.runtime, step);
  const runtimeScript = await assemble(assets.script, step);
  const runtimeModule = await assemble(assets.module, step);
  const wasm = await gunzip(runtime, assets.runtime.expandedBytes);

  const model = await cachedFile(vadModelUrl, speechAssets.silero, step);
  const speech = [];
  for (const lang of SPEECH_LANGS) {
    const file = (name) => cachedFile(new URL(`${lang}/${name}`, speechFolder), speechAssets[lang][name], step);
    speech.push({ lang, encoder: await file("encoder.onnx"), decoder: await file("decoder.onnx"), vocab: await file("vocab.json") });
  }
  return { runtimeScript, runtimeModule, wasm: wasm.buffer, model, speech };
}

// One worker for the whole visit; preparing the model again is slow.
let vadWorker;
let vadReady;
function prepareVoiceModel(onProgress) {
  vadReady ??= (async () => {
    const files = await loadVoiceModel(onProgress);
    vadWorker = new Worker("vad-worker.js?v=2");
    await new Promise((resolve, reject) => {
      vadWorker.onmessage = ({ data }) => (data.error ? reject(new Error(data.error)) : resolve());
      vadWorker.onerror = (event) => reject(new Error(event.message));
      const transfer = [files.wasm, files.model, ...files.speech.flatMap((s) => [s.encoder, s.decoder, s.vocab])];
      vadWorker.postMessage({ type: "prepare", ...files }, transfer);
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

// Finds short phrases (see the constants at the top). Frames are numbered from the
// start, so the worker can cut out the audio of a phrase.
function createPhraseDetector(onPhrase) {
  let speaking = false;
  let quietFrames = 0;
  let phraseFrames = 0;
  let startFrame = 0;
  return {
    reset() {
      speaking = false;
      quietFrames = 0;
      phraseFrames = 0;
    },
    push(probability, frame) {
      if (!speaking) {
        if (probability >= SPEECH_ON) {
          speaking = true;
          startFrame = frame;
          phraseFrames = 1;
          quietFrames = 0;
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
        log(`phrase ${seconds.toFixed(2)} s`);
        if (seconds >= MIN_PHRASE && seconds <= MAX_PHRASE) onPhrase(startFrame, startFrame + phraseFrames, seconds);
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
  let reading = false;     // a phrase is being read by the speech model
  let request = 0;

  // A phrase is read by the speech model; only the question takes the picture.
  const detector = createPhraseDetector((start, end, seconds) => {
    if (reading) return;
    reading = true;
    request += 1;
    vadWorker?.postMessage({ type: "transcribe", start, end, id: request });
    log(`reading a ${seconds.toFixed(2)} s phrase`);
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
    reading = false;
    request += 1;            // answers to earlier phrases are ignored
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
      if (data.texts || data.id) {
        if (data.id !== request) return;
        reading = false;
        if (data.error) {
          log("speech model error:", data.error);
          return;
        }
        log(`heard: ${data.texts.map(({ lang, text }) => `${lang} "${text}"`).join(", ")} (${data.ms} ms)`);
        if (data.texts.some(({ text }) => isQuestion(text))) {
          log("question detected");
          pause();
          onQuestion();
        }
        return;
      }
      if (data.error) {
        log("voice model error:", data.error);
        return;
      }
      data.probs.forEach((probability, i) => { speaking = detector.push(probability, data.first + i); });
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
