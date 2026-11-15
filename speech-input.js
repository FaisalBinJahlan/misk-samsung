"use strict";

const spokenQuestions = [
  "ما هذا", "ما هذه", "وش هذا", "وش هذي", "إيش هذا", "إيش هذي",
  "شنو هذا", "شنو هذي", "وش اسمه", "وش اسمها", "وش اسم هذا", "وش اسم هذي",
  "إيش اسمه", "إيش اسمها", "شنو اسمه", "شنو اسمها", "ما اسم هذا", "ما اسم هذه",
  "وش هذا بالإنجليزي", "إيش هذا بالإنجليزي", "وش اسمه بالإنجليزي", "وش اسمها بالإنجليزي",
  "شنو اسمه بالإنجليزي", "كيف أقول هذا بالإنجليزي", "سمّ هذا بالإنجليزي", "قل لي وش هذا",
  "What is this", "What's this thing", "What's it called", "What's this called",
];

function normalizeQuestion(transcript) {
  return transcript.toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[إأآٱ]/g, "ا")
    .replace(/[\u064B-\u065F\u0670\u0640]/g, "")
    .replace(/[^\u0621-\u064Aa-z\s]/g, " ")
    .trim()
    .replace(/\s+/g, " ")
    .replace(/^(?:(?:طيب|يلا|please) )+/, "")
    .replace(/ (?:لو سمحت|من فضلك|please)$/, "")
    .replace(/^ما(?=هذا|هذه|اسم)/, "ما ")
    .replace(/\bwhats\b/g, "what is");
}

const questionSet = new Set(spokenQuestions.map(normalizeQuestion));

function createSpeechInput({ canListen, onQuestion, onError }) {
  const Recognition = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  const wave = document.getElementById("mic-wave");
  const state = document.getElementById("mic-state");
  const privacy = document.getElementById("mic-privacy");
  const language = document.getElementById("question-language");
  const localPacks = new Set();
  let enabled = false;
  let current;
  let restartTimer;
  let startTimer;

  if (typeof Recognition?.available === "function") {
    for (const option of language.options) {
      Promise.resolve().then(() => Recognition.available({ langs: [option.value], processLocally: true }))
        .then((status) => {
          if (status === "available") localPacks.add(option.value);
          updatePrivacy();
        }).catch(() => {});
    }
  }

  function updatePrivacy() {
    const isLocal = current ? current.processLocally === true : localPacks.has(language.value);
    privacy.textContent = isLocal ? "التعرّف الصوتي على جهازك." : "قد يعالج المتصفح الكلام عبر الإنترنت.";
  }

  function halt() {
    clearTimeout(startTimer);
    clearTimeout(restartTimer);
    wave.dataset.speaking = "false";
    if (!current) return;
    const recognizer = current;
    current = undefined;
    try { recognizer.abort(); } catch {}
  }

  function fail(message) {
    enabled = false;
    halt();
    onError(message);
  }

  function start() {
    clearTimeout(restartTimer);
    if (!enabled || !canListen() || current) return;
    if (!Recognition) {
      fail("متصفحك ما يدعم المايك للتعرّف الصوتي. استخدم زر «ماهذا؟».");
      return;
    }

    let recognizer;
    try { recognizer = new Recognition(); } catch {
      fail("متصفحك ما قدر يجهّز التعرّف الصوتي. استخدم زر «ماهذا؟».");
      return;
    }
    current = recognizer;
    recognizer.lang = language.value;
    if ("processLocally" in recognizer && localPacks.has(recognizer.lang)) recognizer.processLocally = true;
    recognizer.continuous = false;
    recognizer.interimResults = false;
    recognizer.maxAlternatives = 2;
    updatePrivacy();
    state.textContent = "جاري تشغيل المايك…";

    recognizer.onstart = () => {
      if (current !== recognizer) return;
      clearTimeout(startTimer);
      state.textContent = recognizer.lang === "en-US" ? "قل: What is this?" : "أسمعك، قل «ما هذا؟»";
    };
    recognizer.onspeechstart = () => {
      if (current === recognizer) wave.dataset.speaking = "true";
    };
    recognizer.onspeechend = () => {
      if (current === recognizer) wave.dataset.speaking = "false";
    };
    recognizer.onresult = (event) => {
      if (current !== recognizer || !canListen()) return;
      for (const result of Array.from(event.results).slice(event.resultIndex)) {
        if (result.isFinal && Array.from(result).some(({ transcript }) => questionSet.has(normalizeQuestion(transcript)))) {
          halt();
          onQuestion();
          return;
        }
      }
    };
    recognizer.onerror = ({ error }) => {
      if (current !== recognizer || error === "no-speech" || error === "aborted") return;
      const denied = error === "not-allowed" || error === "service-not-allowed";
      fail(denied
        ? "المايك يحتاج إذن من المتصفح. تقدر تكمل بزر «ماهذا؟»."
        : "التعرّف الصوتي تعذّر. تقدر تكمل بزر «ماهذا؟» أو تجرّب المايك مرة ثانية.");
    };
    recognizer.onend = () => {
      if (current !== recognizer) return;
      halt();
      restartTimer = setTimeout(start, 350);
    };
    startTimer = setTimeout(() => {
      if (current === recognizer) fail("المايك ما بدأ. جرّب مرة ثانية أو استخدم زر «ماهذا؟».");
    }, 10000);
    try { recognizer.start(); } catch {
      fail("ما قدر المتصفح يشغّل المايك. تقدر تستخدم زر «ماهذا؟».");
    }
  }

  language.addEventListener("change", () => {
    halt();
    updatePrivacy();
    start();
  });

  return {
    enable() { enabled = true; start(); },
    disable() { enabled = false; halt(); },
    refresh() {
      if (enabled && canListen()) start();
      else {
        halt();
        if (enabled) state.textContent = "المايك متوقف مؤقتًا…";
      }
    },
  };
}
