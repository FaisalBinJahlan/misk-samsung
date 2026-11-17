"use strict";

// Voice question for the camera ("ماهذا؟").
//
// The browser's SpeechRecognition listens for an actual question ("ماهذا؟",
// "وش هذا؟", "ايش ذا؟" ...). Only a question takes a picture; coughs, a TV or
// other talk do not. On-device recognition is used when the browser offers it;
// otherwise the browser's online speech service is used (the audio goes to the
// browser maker for recognition; camera images never leave the device).
// ?speech=local refuses the online service. When no recognition is possible the
// app switches back to the "ماهذا؟" button: it never reacts to noise alone. app.js only uses
// createSpeechInput({ canListen, onQuestion, onError }) -> { enable, disable, refresh }.

const QUESTION_LANG = "ar-SA";

// Arabic: remove diacritics and tatweel, unify letter forms, keep letters and spaces.
function normalizeArabic(text) {
  return text
    .toLowerCase()
    .replace(/[‘’ʼ]/g, "'")
    .replace(/[ً-ٰٟـ]/g, "")
    .replace(/[أإآٱ]/g, "ا")
    .replace(/ى/g, "ي")
    .replace(/ة/g, "ه")
    .replace(/[^\p{L}\p{N}\s']/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Question forms we accept. "ذا" / "ذي" on their own are only accepted after
// a dialect word (وش، ايش، شو), so "ماذا تفعل" does not count as a question.
const QUESTION_PATTERNS = [
  /(?:^| )(?:ما|ماذا|وش|وشو|ايش|اش|شو|شنو|شنهو) ?(?:هو |هي )?(?:هذا|هذه|هذي|هاذا|هاذي)(?: |$)/,
  /(?:^| )(?:وش|وشو|ايش|اش|شو) ?(?:ذا|ذي)(?: |$)/,
  /(?:^| )(?:what is|what's|whats|what) (?:this|that|it)(?: |$)/,
];

function isQuestion(transcript) {
  const text = normalizeArabic(transcript);
  return QUESTION_PATTERNS.some((pattern) => pattern.test(text));
}

const debugSpeech = new URLSearchParams(location.search).has("debug");
function log(...parts) {
  if (debugSpeech) console.info("[speech]", ...parts);
}

// Microphone level meter: moves the bars so the child can see they are heard.
function createLevelMeter({ canListen, onError }) {
  const wave = document.getElementById("mic-wave");
  const levelBars = [...wave.children];
  const AudioEngine = window.AudioContext ?? window.webkitAudioContext;
  const barWeights = [0.35, 0.62, 0.83, 1, 0.83, 0.62, 0.35];
  let current;
  let frame;
  let lastSample;
  let noise = 0.004;
  let level = 0;

  function renderLevel(next) {
    for (const [i, bar] of levelBars.entries()) {
      bar.style.transform = `scaleY(${0.16 + next * 0.84 * barWeights[i]})`;
    }
  }

  function reset() {
    cancelAnimationFrame(frame);
    frame = undefined;
    lastSample = undefined;
    level = 0;
    renderLevel(0);
  }

  function disable() {
    reset();
    const session = current;
    current = undefined;
    if (!session) return;
    session.track?.stop();
    session.source?.disconnect();
    session.context.close().catch(() => {});
  }

  function fail(message) {
    disable();
    onError(message);
  }

  function sample(now) {
    frame = undefined;
    if (!current?.track?.enabled || !canListen()) return;
    const elapsed = now - (lastSample ?? now);
    if (lastSample === undefined || elapsed >= 30) {
      lastSample = now;
      current.analyser.getFloatTimeDomainData(current.samples);
      let total = 0;
      for (const value of current.samples) total += value * value;
      const rms = Math.sqrt(total / current.samples.length);
      const threshold = Math.max(0.012, noise * 3.1);
      const audible = rms > threshold;
      const visual = audible ? Math.min(1, Math.sqrt((rms - threshold) * 16)) : 0;
      level += (visual - level) * (visual > level ? 0.7 : 0.25);
      renderLevel(level);

      if (!audible) noise += (rms - noise) * 0.04;
    }
    frame = requestAnimationFrame(sample);
  }

  function refresh() {
    const track = current?.track;
    if (!track || !current.analyser) return;
    const listening = canListen();
    track.enabled = listening;
    if (!listening) reset();
    else if (frame === undefined) frame = requestAnimationFrame(sample);
  }

  async function enable() {
    if (!AudioEngine || !navigator.mediaDevices?.getUserMedia) {
      fail("متصفحك ما قدر يشغّل المايك. استخدم زر «ماهذا؟».");
      return;
    }
    disable();
    let session;
    try {
      const context = new AudioEngine();
      session = { context };
      current = session;
      const running = context.resume().catch(() => false);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (current !== session) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      [session.track] = stream.getAudioTracks();
      await running;
      if (current !== session) return;
      if (context.state !== "running" || !session.track) throw new Error("Microphone unavailable");
      session.track.addEventListener("ended", () => {
        if (current === session) fail("توقف المايك. جرّب تشغيله مرة ثانية.");
      }, { once: true });
      session.source = context.createMediaStreamSource(stream);
      session.analyser = context.createAnalyser();
      session.analyser.fftSize = 1024;
      session.samples = new Float32Array(session.analyser.fftSize);
      session.source.connect(session.analyser);
      noise = 0.004;
      reset();
      refresh();
    } catch (error) {
      if (session && current !== session) return;
      fail(error.name === "NotAllowedError"
        ? "اسمح بالمايك من المتصفح أو استخدم زر «ماهذا؟»."
        : "تعذّر تشغيل المايك. جرّب مرة ثانية أو استخدم زر «ماهذا؟».");
    }
  }

  return { enable, disable, refresh };
}

// Decide how speech can be recognised on this browser:
// "local" (on the device, preferred), "cloud" (the browser's online service)
// or "" (no speech recognition: the button is used instead).
// Opening the page with ?speech=local refuses the online service.
async function chooseRecognitionMode(Recognition) {
  if (!Recognition) return "";
  const localOnly = new URLSearchParams(location.search).get("speech") === "local";
  if (typeof Recognition.available !== "function") return localOnly ? "" : "cloud";
  try {
    const options = { langs: [QUESTION_LANG], processLocally: true };
    let status = await Recognition.available(options);
    log("on-device availability:", status);
    if ((status === "downloadable" || status === "downloading") && typeof Recognition.install === "function") {
      const installed = await Recognition.install(options);
      log("on-device install:", installed);
      if (installed) status = "available";
    }
    if (status === "available") return "local";
    if (localOnly) return "";
    const online = await Recognition.available({ langs: [QUESTION_LANG], processLocally: false });
    log("online availability:", online);
    return online === "unavailable" ? "" : "cloud";
  } catch (error) {
    log("availability check failed:", error);
    return localOnly ? "" : "cloud";
  }
}

function createSpeechInput({ canListen, onQuestion, onError }) {
  const state = document.getElementById("mic-state");
  const Recognition = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  let mode = "";           // "speech" while listening
  let token = 0;           // changes on every enable/disable, so stale callbacks are ignored
  let recognition;
  let running = false;
  let meter;               // microphone level bars
  let restartTimer;

  function stopRecognition() {
    clearTimeout(restartTimer);
    if (recognition && running) {
      try { recognition.abort(); } catch {}
    }
    running = false;
  }

  function disable() {
    token += 1;
    stopRecognition();
    recognition = undefined;
    meter?.disable();
    meter = undefined;
    mode = "";
  }


  function startRecognition() {
    if (!recognition || running || !canListen()) return;
    try {
      recognition.start();
      running = true;
      state.textContent = "أسمعك… اسأل: ماهذا؟";
    } catch (error) {
      log("start failed:", error);
    }
  }

  function refresh() {
    if (mode !== "speech") return;
    meter?.refresh();
    if (canListen()) {
      startRecognition();
    } else {
      stopRecognition();
      state.textContent = "المايك متوقف مؤقتًا";
    }
  }

  async function enable() {
    disable();
    const myToken = token;
    state.textContent = "جاري تشغيل المايك…";
    const recognitionMode = await chooseRecognitionMode(Recognition);
    if (myToken !== token) return;

    if (!recognitionMode) {
      log("no speech recognition available");
      disable();
      onError("التعرّف على الكلام غير متاح في هذا المتصفح. استخدم زر «ماهذا؟».");
      return;
    }

    mode = "speech";
    recognition = new Recognition();
    recognition.lang = QUESTION_LANG;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;
    if (recognitionMode === "local") recognition.processLocally = true;
    log("engine:", recognitionMode, QUESTION_LANG);

    let heardAt;
    recognition.onspeechstart = () => { heardAt = performance.now(); };
    recognition.onresult = (event) => {
      if (myToken !== token) return;
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        for (let j = 0; j < result.length; j++) {
          const transcript = result[j].transcript;
          log(result.isFinal ? "final:" : "interim:", transcript);
          if (isQuestion(transcript)) {
            const waited = heardAt ? Math.round(performance.now() - heardAt) : "?";
            log(`question recognised after ${waited} ms:`, transcript);
            stopRecognition();
            onQuestion();
            return;
          }
        }
      }
    };
    recognition.onerror = (event) => {
      if (myToken !== token) return;
      // "no-speech" and "aborted" happen all the time while waiting: not real errors.
      if (event.error !== "no-speech" && event.error !== "aborted") log("error:", event.error);
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        disable();
        onError("اسمح بالمايك من المتصفح أو استخدم زر «ماهذا؟».");
      } else if (event.error === "language-not-supported" || event.error === "network") {
        // Speech cannot be recognised here: never fall back to reacting to any noise.
        disable();
        onError(event.error === "network"
          ? "التعرّف على الكلام يحتاج اتصال إنترنت. استخدم زر «ماهذا؟»."
          : "التعرّف على الكلام بالعربية غير متاح هنا. استخدم زر «ماهذا؟».");
      } else if (event.error === "audio-capture") {
        // Some phones allow only one microphone user: drop the level bars and retry.
        meter?.disable();
        meter = undefined;
      }
      // "no-speech" and "aborted" need nothing: onend restarts listening.
    };
    recognition.onend = () => {
      running = false;
      if (myToken !== token || mode !== "speech") return;
      // Chrome ends a session after a period of silence; keep listening while allowed.
      restartTimer = setTimeout(() => { if (myToken === token) startRecognition(); }, 250);
    };

    // Level bars only, so the child can see the microphone hears them.
    meter = createLevelMeter({ canListen, onError });
    meter.enable();
    startRecognition();
  }

  return { enable, disable, refresh };
}
