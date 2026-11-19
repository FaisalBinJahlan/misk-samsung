"use strict";

// Voice question: takes a picture only when the child asks "ماهذا؟" (any dialect, or English).
// app.js uses createSpeechInput({ canListen, onQuestion, onError }) -> { enable, disable, refresh }.

const QUESTION_LANG = "ar-SA";

// Lowercases, removes diacritics and unifies letter forms (أ/إ/آ → ا, ى → ي, ة → ه).
function normalizeArabic(text) {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019\u02BC]/g, "'")
    .replace(/[ً-ٰٟـ]/g, "")
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

const debugSpeech = new URLSearchParams(location.search).has("debug");
function log(...parts) {
  if (debugSpeech) console.info("[speech]", ...parts);
}

// iPhone/iPad: long sessions are unreliable, so they use short self-restarting ones.
const isAppleMobile = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

// Mic bars driven by the recognizer's sound events (opening the mic twice breaks phones).
function createListeningBars() {
  const bars = [...(document.getElementById("mic-wave")?.children ?? [])];
  const weights = [0.35, 0.62, 0.83, 1, 0.83, 0.62, 0.35];
  let timer;
  let hearing = false;
  let level = 0;

  function draw() {
    const target = hearing ? 0.45 + Math.random() * 0.55 : 0;
    level += (target - level) * 0.5;
    for (const [i, bar] of bars.entries()) {
      bar.style.transform = `scaleY(${0.16 + level * 0.84 * weights[i]})`;
    }
    if (!hearing && level < 0.02) {
      clearInterval(timer);
      timer = undefined;
    }
  }

  return {
    set(on) {
      hearing = on;
      if (!timer) timer = setInterval(draw, 90);
    },
    stop() {
      hearing = false;
      level = 0;
      clearInterval(timer);
      timer = undefined;
      for (const bar of bars) bar.style.transform = "scaleY(0.16)";
    },
  };
}

// Returns "local" (on-device), "cloud" (browser's online service) or "" (none).
// ?speech=local refuses the online service.
async function chooseRecognitionMode(Recognition, lang) {
  if (!Recognition) return "";
  const localOnly = new URLSearchParams(location.search).get("speech") === "local";
  if (typeof Recognition.available !== "function") return localOnly ? "" : "cloud";
  try {
    const options = { langs: [lang], processLocally: true };
    let status = await Recognition.available(options);
    log("on-device availability:", status);
    if ((status === "downloadable" || status === "downloading") && typeof Recognition.install === "function") {
      const installed = await Recognition.install(options);
      log("on-device install:", installed);
      if (installed) status = "available";
    }
    if (status === "available") return "local";
    if (localOnly) return "";
    const online = await Recognition.available({ langs: [lang], processLocally: false });
    log("online availability:", online);
    return online === "unavailable" ? "" : "cloud";
  } catch (error) {
    log("availability check failed:", error);
    return localOnly ? "" : "cloud";
  }
}

const TEXT = {
  listening: "أسمعك… اسأل: ماهذا؟ أو What is this?",
  paused: "المايك متوقف مؤقتًا",
  starting: "جاري تشغيل المايك…",
  tap: "اضغط هنا ثم اسأل: ماهذا؟",
  tapListening: "أسمعك… اسأل الآن",
  unsupported: "متصفحك ما يدعم السؤال بالصوت. افتح الرابط في Chrome أو Safari، أو استخدم زر «ماهذا؟».",
  denied: isAppleMobile
    ? "اسمح بالمايك، وفعّل Siri والإملاء من الإعدادات، أو استخدم زر «ماهذا؟»."
    : "اسمح بالمايك من المتصفح أو استخدم زر «ماهذا؟».",
  offline: "السؤال بالصوت يحتاج اتصال إنترنت. استخدم زر «ماهذا؟».",
  noArabic: "التعرّف على الكلام بالعربية غير متاح هنا. استخدم زر «ماهذا؟».",
};

// "handsfree" on every device; "tap" (tap the mic panel, then ask) only if listening keeps failing.
function createSpeechInput({ canListen, onQuestion, onError }) {
  const state = document.getElementById("mic-state");
  const panel = document.getElementById("mic-panel");
  const Recognition = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  const bars = createListeningBars();
  let mode = "";            // "", "handsfree", "tap"
  let engine = "";          // "local" or "cloud"
  let token = 0;            // ignores events from old sessions
  let recognition;
  let running = false;
  let restartTimer;
  let watchdog;
  let quickEnds = 0;        // sessions in a row that ended at once, hearing nothing
  let speechUnlocked = false;

  function setText(text) {
    if (state) state.textContent = text;
  }

  function idleText() {
    if (!canListen()) return TEXT.paused;
    return mode === "tap" ? TEXT.tap : TEXT.listening;
  }

  function stopSession() {
    clearTimeout(restartTimer);
    clearTimeout(watchdog);
    if (recognition && running) {
      try { recognition.abort(); } catch {}
    }
    running = false;
    bars.set(false);
  }

  function disable() {
    token += 1;
    stopSession();
    recognition = undefined;
    bars.stop();
    mode = "";
    panel?.removeAttribute("role");
    panel?.removeAttribute("tabindex");
    if (panel) panel.style.cursor = "";
  }

  function useTapMode(reason) {
    if (mode === "tap") return;
    log("switching to tap-to-ask:", reason);
    stopSession();
    mode = "tap";
    if (panel) {
      panel.setAttribute("role", "button");
      panel.setAttribute("tabindex", "0");
      panel.style.cursor = "pointer";
    }
    setText(idleText());
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

  function scheduleRestart(myToken) {
    if (mode !== "handsfree") return;
    if (quickEnds >= 4) {
      useTapMode("listening kept stopping by itself");
      return;
    }
    // Back off after each quick failure.
    const delay = Math.min(250 * 2 ** quickEnds, 4000);
    restartTimer = setTimeout(() => { if (myToken === token) startSession(); }, delay);
  }

  function startSession() {
    if (!mode || running || !canListen()) return;
    const myToken = token;
    // New recognizer per session (iPhones misbehave when one is reused).
    const rec = new Recognition();
    rec.lang = QUESTION_LANG;
    rec.continuous = mode === "handsfree" && !isAppleMobile;
    rec.interimResults = true;
    rec.maxAlternatives = 5;
    if (engine === "local") rec.processLocally = true;
    recognition = rec;

    let startedAt = performance.now();
    let heardSound = false;
    let heardAt;

    rec.onsoundstart = () => { heardSound = true; bars.set(true); };
    rec.onspeechstart = () => { heardSound = true; heardAt = performance.now(); bars.set(true); };
    rec.onsoundend = () => bars.set(false);
    rec.onspeechend = () => bars.set(false);

    rec.onresult = (event) => {
      if (myToken !== token || rec !== recognition) return;
      heardSound = true;
      // Check every alternative, plus the last pieces joined ("وش" + "هذا" may arrive apart).
      const candidates = [];
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        log(result.isFinal ? "final:" : "interim:", result[0].transcript);
        for (let j = 0; j < result.length; j++) candidates.push(result[j].transcript);
      }
      const recent = [];
      for (let i = Math.max(0, event.results.length - 3); i < event.results.length; i++) {
        recent.push(event.results[i][0].transcript);
      }
      candidates.push(recent.join(" "));

      const heard = candidates.find(isQuestion);
      if (heard) {
        const waited = heardAt ? Math.round(performance.now() - heardAt) : "?";
        log(`question recognised after ${waited} ms:`, heard);
        quickEnds = 0;
        stopSession();
        setText(idleText());
        onQuestion();
      }
    };

    rec.onerror = (event) => {
      if (myToken !== token || rec !== recognition) return;
      if (event.error !== "no-speech" && event.error !== "aborted") log("error:", event.error);
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        disable();
        onError(TEXT.denied);
      } else if (event.error === "network") {
        disable();
        onError(TEXT.offline);
      } else if (event.error === "language-not-supported") {
        disable();
        onError(TEXT.noArabic);
      }
    };

    rec.onend = () => {
      if (myToken !== token || rec !== recognition) return;
      running = false;
      clearTimeout(watchdog);
      bars.set(false);
      const lasted = performance.now() - startedAt;
      quickEnds = !heardSound && lasted < 1500 ? quickEnds + 1 : 0;
      if (mode === "handsfree") scheduleRestart(myToken);
      else setText(idleText());
    };

    try {
      rec.start();
      running = true;
      startedAt = performance.now();
      setText(mode === "tap" ? TEXT.tapListening : TEXT.listening);
      // Some phones freeze silently: reset after a timeout.
      watchdog = setTimeout(() => {
        if (myToken === token && rec === recognition && running) {
          log("no answer from the recognizer, restarting");
          try { rec.abort(); } catch {}
        }
      }, mode === "tap" ? 10000 : 60000);
    } catch (error) {
      log("start failed:", error);
      running = false;
      quickEnds += 1;
      if (mode === "handsfree") scheduleRestart(myToken);
      else setText(idleText());
    }
  }

  function onPanelTap() {
    if (mode !== "tap") return;
    unlockSpeech();
    if (!running && canListen()) startSession();
  }
  panel?.addEventListener("click", onPanelTap);
  panel?.addEventListener("keydown", (event) => {
    if (mode === "tap" && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      onPanelTap();
    }
  });

  function refresh() {
    if (!mode) return;
    if (!canListen()) {
      stopSession();
      setText(TEXT.paused);
    } else if (mode === "handsfree") {
      if (!running) startSession();
    } else if (!running) {
      setText(idleText());
    }
  }

  async function enable() {
    disable();
    const myToken = token;
    unlockSpeech();
    setText(TEXT.starting);
    if (!Recognition) {
      log("this browser has no speech recognition");
      onError(TEXT.unsupported);
      return;
    }
    engine = await chooseRecognitionMode(Recognition, QUESTION_LANG);
    if (myToken !== token) return;
    if (!engine) {
      log("no speech recognition available");
      onError(TEXT.unsupported);
      return;
    }
    quickEnds = 0;
    log("engine:", engine, QUESTION_LANG, isAppleMobile ? "(hands-free, short sessions)" : "(hands-free)");
    mode = "handsfree";
    startSession();
  }

  return { enable, disable, refresh };
}
