"use strict";

const experience = document.getElementById("experience");
const camera = document.getElementById("camera");
const viewfinder = document.getElementById("viewfinder");
const action = document.getElementById("action");
const stop = document.getElementById("stop");
const status = document.getElementById("status");
const result = document.getElementById("result");
const word = document.getElementById("word");
const sentence = document.getElementById("sentence");
const replay = document.getElementById("replay");
const voiceSettings = document.getElementById("voice-settings");
const voiceDialog = document.getElementById("voice-dialog");
const voiceSelect = document.getElementById("voice-select");
const voicePreview = document.getElementById("voice-preview");
const voiceNote = document.getElementById("voice-note");
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const speech = window.speechSynthesis;
const voiceStorageKey = "misk-english-voice";
let selectedVoice = "";
let draftVoice = "";
try { selectedVoice = localStorage.getItem(voiceStorageKey) ?? ""; } catch {}

let stream;
let cameraSession = 0;
let busy = false;
let spokenWord = "";
let expandedWorker;
let cancelExpanded;
const expandedFolder = new URL("vendor/expanded-lite/", document.baseURI);
const expandedCacheName = "misk-expanded-model-v5";

// GitHub Pages lets browsers keep the page for a while, so phones can keep
// running an old version. Compare with version.json and reload when it changed.
const appVersion = "13";
async function checkForUpdate() {
  try {
    const response = await fetch(`version.json?t=${Date.now()}`, { cache: "no-store" });
    const { version } = await response.json();
    if (!version || version === appVersion || busy) return;
    const key = `misk-reloaded-${version}`;
    if (sessionStorage.getItem(key)) return;
    sessionStorage.setItem(key, "1");
    location.replace(`${location.pathname}?v=${version}`);
  } catch {}
}
checkForUpdate();
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") checkForUpdate();
});

function downloadExpandedPart(url, signal, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url);
    request.responseType = "arraybuffer";
    const abort = () => request.abort();
    let timer;
    let timedOut = false;
    function resetTimer() {
      clearTimeout(timer);
      timer = setTimeout(() => {
        timedOut = true;
        request.abort();
      }, 20000);
    }
    const failed = () => reject(new Error(timedOut ? "Download stalled" : "Download interrupted"));
    request.onprogress = (event) => {
      resetTimer();
      onProgress(event.loaded);
    };
    request.onload = () => {
      if (request.status === 200 && request.response && !request.getResponseHeader("content-type")?.includes("text/html")) {
        resolve(request.response);
      } else {
        failed();
      }
    };
    request.onabort = request.onerror = failed;
    request.onloadend = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
    };
    if (signal.aborted) return failed();
    signal.addEventListener("abort", abort, { once: true });
    resetTimer();
    request.send();
  });
}

async function downloadExpanded(session) {
  const controller = new AbortController();
  const { signal } = controller;
  cancelExpanded = () => controller.abort();
  const cache = await window.caches?.open(expandedCacheName).catch(() => null);
  // Reclaim space from bundles this browser cached for older versions.
  if (cache) {
    await window.caches.delete("misk-expanded-model-v1").catch(() => {});
    await window.caches.delete("misk-expanded-model-v2").catch(() => {});
    await window.caches.delete("misk-expanded-model-v3").catch(() => {});
    await window.caches.delete("misk-expanded-model-v4").catch(() => {});
  }
  const progress = new Map();
  let total = 0;
  let retrying = 0;
  let lastMessage;
  function report() {
    if (session !== cameraSession) return;
    const sum = [...progress.values()].reduce((a, b) => a + b, 0);
    const percent = total ? Math.min(99, Math.floor(100 * sum / total)) : 0;
    const caption = retrying ? "جاري إعادة محاولة الاتصال" : "جاري تحميل نموذج التعرّف";
    const message = `${caption}… ${percent}٪`;
    if (message !== lastMessage) {
      lastMessage = message;
      status.textContent = message;
    }
  }
  async function part(name, expectedBytes) {
    const url = new URL(name, expandedFolder).href;
    const cached = await cache?.match(url).catch(() => null);
    let data = await cached?.arrayBuffer().catch(() => null);
    if (!data || (expectedBytes && data.byteLength !== expectedBytes)) {
      let lastError;
      for (let attempt = 0; attempt < 3 && !signal.aborted; attempt++) {
        if (attempt) {
          retrying++;
          report();
        }
        try {
          const requestUrl = new URL(url);
          if (attempt) requestUrl.searchParams.set("retry", String(Date.now()));
          data = await downloadExpandedPart(requestUrl.href, signal, (loaded) => {
            if (expectedBytes) progress.set(name, loaded);
            report();
          });
          if (expectedBytes && data.byteLength !== expectedBytes) throw new Error("Incomplete model part");
          if (expectedBytes) await cache?.put(url, new Response(data)).catch(() => {});
          lastError = null;
          break;
        } catch (error) {
          data = null;
          progress.delete(name);
          lastError = error;
        } finally {
          if (attempt) retrying--;
          report();
        }
      }
      if (!data || signal.aborted) throw lastError ?? new Error("Download canceled");
    }
    if (expectedBytes) {
      progress.set(name, data.byteLength);
      report();
    }
    return new Uint8Array(data);
  }
  async function assemble(entry) {
    const buffer = new Uint8Array(entry.bytes);
    let offset = 0;
    for (const { name, bytes } of entry.parts) {
      const piece = await part(name, bytes);
      buffer.set(piece, offset);
      offset += piece.length;
    }
    if (offset !== entry.bytes) throw new Error("Incomplete expanded model");
    return buffer;
  }

  try {
    report();
    const manifest = await part("assets.json");
    const assets = JSON.parse(new TextDecoder().decode(manifest));
    total = Object.values(assets).reduce((bytes, asset) => bytes + asset.bytes, 0);
    // Keep the number of competing requests small on the private tunnel.
    const [[model, index], [wasm, vocabulary, runtimeScript, runtimeModule]] = await Promise.all([
      (async () => [await assemble(assets.vision), await assemble(assets.index)])(),
      (async () => [
        await assemble(assets.runtime), await assemble(assets.labels),
        await assemble(assets.script), await assemble(assets.module),
      ])(),
    ]);
    if (session !== cameraSession) throw new Error("Camera session ended");
    const lengths = [assets.vision, assets.runtime, assets.labels, assets.index].map((entry) => entry.expandedBytes);
    return { model, wasm, vocabulary, index, runtimeScript, runtimeModule, lengths };
  } finally {
    controller.abort();
    if (session === cameraSession) cancelExpanded = undefined;
  }
}

function requestExpanded(message, session) {
  expandedWorker ??= new Worker("expanded-worker.js?v=11");
  const worker = expandedWorker;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => fail(new Error("انتهت مهلة تجهيز التعرّف.")), 150000);
    const fail = (error) => {
      clearTimeout(timer);
      worker.terminate();
      if (expandedWorker === worker) expandedWorker = undefined;
      cancelExpanded = undefined;
      reject(error);
    };
    cancelExpanded = () => fail(new Error("Camera session ended"));
    worker.onerror = (event) => fail(new Error(event.message));
    worker.onmessage = ({ data }) => {
      if (data.error) {
        fail(new Error(data.error));
      } else {
        clearTimeout(timer);
        cancelExpanded = undefined;
        resolve(data);
      }
    };
    const buffers = Object.values(message).filter(ArrayBuffer.isView).map((value) => value.buffer);
    try {
      worker.postMessage(message, buffers);
    } catch (error) {
      fail(error);
    }
  });
}

async function recognizeExpanded(session) {
  const pixels = context.getImageData(0, 0, 256, 256).data;
  if (!expandedWorker) {
    const assets = await downloadExpanded(session);
    if (session !== cameraSession) return "";
    status.textContent = "اكتمل التحميل. جاري تجهيز التعرّف…";
    await requestExpanded({ type: "prepare", ...assets }, session);
  }
  if (session !== cameraSession) return "";
  status.textContent = "جاري التعرّف…";
  const { label } = await requestExpanded({ type: "recognize", pixels }, session);
  return label;
}

function availableEnglishVoices() {
  return speech.getVoices().filter((voice) => /^en(?:[-_]|$)/i.test(voice.lang));
}

function refreshVoiceOptions() {
  const voices = availableEnglishVoices();
  voiceSelect.replaceChildren(new Option("تلقائي", ""));
  for (const voice of voices) {
    voiceSelect.add(new Option(`${voice.name} · ${voice.lang}`, voice.voiceURI));
  }
  voiceSelect.value = voices.some((voice) => voice.voiceURI === draftVoice) ? draftVoice : "";
  voiceNote.hidden = voices.length > 0;
  voiceNote.textContent = "ما ظهرت أصوات إنجليزية إضافية من المتصفح. جرّب الصوت التلقائي.";
}

function say(text, { auto = false, voiceId = selectedVoice, preview = false } = {}) {
  if (!speech || !window.SpeechSynthesisUtterance) return;
  speech.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voices = availableEnglishVoices();
  utterance.voice = voices.find((voice) => voice.voiceURI === voiceId)
    ?? voices.find((voice) => voice.localService && voice.lang.toLowerCase() === "en-us")
    ?? voices.find((voice) => voice.localService)
    ?? voices.find((voice) => voice.lang.toLowerCase() === "en-us")
    ?? null;
  utterance.lang = utterance.voice?.lang ?? "en-US";
  utterance.rate = 0.85;
  utterance.onerror = (event) => {
    if (event.error === "canceled" || event.error === "interrupted") return;
    if (preview) {
      if (!voiceDialog.open) return;
      voiceNote.hidden = false;
      voiceNote.textContent = "تعذّر تشغيل هذا الصوت. جرّب صوتًا آخر.";
    } else if (auto) status.textContent = "اضغط زر الصوت لسماع الكلمة.";
    else status.textContent = "تعذّر تشغيل الصوت. الكلمة ظاهرة على الشاشة.";
  };
  speech.speak(utterance);
}

function closeCamera(message = "شغّل الكاميرا ووجّهها نحو غرض واحد.") {
  cameraSession += 1;
  cancelExpanded?.();
  cancelExpanded = undefined;
  if (expandedWorker) {
    expandedWorker.terminate();
    expandedWorker = undefined;
  }
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  camera.srcObject = null;
  experience.classList.remove("camera-on");
  stop.hidden = true;
  result.hidden = true;
  spokenWord = "";
  speech?.cancel();
  status.textContent = message;
  action.textContent = "شغّل الكاميرا";
  action.disabled = false;
  busy = false;
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "افتح رابط HTTPS من متصفح الجوال لتشغيل الكاميرا.";
    return;
  }

  const session = ++cameraSession;
  action.disabled = true;
  status.textContent = "اسمح للصفحة باستخدام الكاميرا.";

  try {
    const requestedStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (session !== cameraSession) {
      requestedStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = requestedStream;
    const [track] = stream.getVideoTracks();
    track.addEventListener("ended", () => {
      if (session === cameraSession) closeCamera("توقفت الكاميرا. شغّلها مرة ثانية.");
    });
    camera.srcObject = stream;
    await camera.play();
    if (session !== cameraSession) return;
  } catch (error) {
    if (session !== cameraSession) return;
    closeCamera(error.name === "NotAllowedError"
      ? "اسمح بالكاميرا في إعدادات المتصفح، ثم جرّب مرة ثانية."
      : "تعذّر فتح الكاميرا. تأكد أنها متاحة وجرّب مرة ثانية.");
    return;
  }

  experience.classList.add("camera-on");
  stop.hidden = false;
  action.textContent = "ماهذا؟";
  status.textContent = "حط الغرض داخل الإطار واضغط الزر.";
  action.disabled = false;
}

// Crop exactly the on-screen square, so only what is inside the frame is read.
function takeSnapshot() {
  const width = camera.videoWidth;
  const height = camera.videoHeight;
  const preview = camera.getBoundingClientRect();
  const target = viewfinder.getBoundingClientRect();
  const scale = Math.max(preview.width / width, preview.height / height);
  const x = (target.left - preview.left + (width * scale - preview.width) / 2) / scale;
  const y = (target.top - preview.top + (height * scale - preview.height) / 2) / scale;
  const side = Math.min(target.width / scale, width, height);
  const left = Math.min(Math.max(0, x), width - side);
  const top = Math.min(Math.max(0, y), height - side);
  canvas.width = 256;
  canvas.height = 256;
  context.drawImage(camera, left, top, side, side, 0, 0, 256, 256);
}

const pluralWords = new Set([
  "AirPods", "binoculars", "biscuits", "blocks", "boots", "boxing gloves", "bubbles", "chickpeas", "chips",
  "chopsticks", "colored pencils", "crackers", "dates", "dice", "earbuds", "earrings", "flip-flops",
  "flippers", "french fries", "gloves", "glasses", "goggles", "grapes", "headphones", "high heels",
  "ice skates", "lentils", "lungs", "marbles", "measuring spoons", "mittens", "noodles", "nuts",
  "oats", "overalls", "pajamas", "pants", "peas", "playing cards", "pliers", "raisins",
  "roller skates", "safety goggles", "sandals", "scissors", "shorts", "skis", "slippers", "sneakers",
  "stairs", "sunglasses", "tongs", "window blinds", "ballet shoes", "beans", "trousers", "jeans",
]);
const massWords = new Set([
  "asparagus", "basil", "bamboo", "bread", "broccoli", "butter", "caramel", "cardamom", "cauliflower",
  "cereal", "chalk", "cheese", "chewing gum", "chocolate", "clay", "coral", "cotton candy",
  "falafel", "fire", "flour", "fruit salad", "garlic", "glue", "gold", "grass",
  "hay", "honey", "hummus", "ice", "jam", "jelly", "juice", "ketchup",
  "lava", "lemonade", "lettuce", "lightning", "meat", "milk", "mint", "mud",
  "mustard", "oil", "okra", "origami", "paint", "paper", "parsley", "pasta", "pita bread", "popcorn",
  "porridge", "rice", "salad", "salt", "sand", "sesame", "shampoo", "snow",
  "soap", "soda", "soil", "soup", "spaghetti", "spinach", "sugar", "sushi",
  "tape", "tea", "toast", "toilet paper", "toothpaste", "water", "wheat", "yarn",
  "yogurt",
]);
const theWords = new Set([
  "earth", "moon", "sea", "sky", "solar system", "sun",
]);

function englishSentence(label) {
  if (pluralWords.has(label)) return `These are ${label}.`;
  if (massWords.has(label)) return `This is ${label}.`;
  if (theWords.has(label)) return `This is the ${label}.`;
  const vowelSound = /^(hour|honest)/.test(label)
    || (/^[aeiou]/.test(label) && !/^(one|uni(?:[fv]|corn|cycle)|use|usb|eu)/.test(label));
  const article = vowelSound ? "an" : "a";
  return `This is ${article} ${label}.`;
}

async function recognize() {
  if (busy || !stream) return;
  if (!camera.videoWidth || camera.readyState < 2) {
    status.textContent = "انتظر حتى تظهر الكاميرا وجرّب مرة ثانية.";
    return;
  }
  const session = cameraSession;
  busy = true;
  action.disabled = true;
  result.hidden = true;
  spokenWord = "";
  speech?.cancel();
  status.textContent = "جاري التعرّف…";

  try {
    takeSnapshot();
    const label = await recognizeExpanded(session);
    if (session !== cameraSession) return;
    if (!label) {
      status.textContent = "قرّب الغرض داخل الإطار وجرّب مرة ثانية.";
      return;
    }
    spokenWord = label;
    word.textContent = spokenWord[0].toUpperCase() + spokenWord.slice(1);
    sentence.textContent = englishSentence(spokenWord);
    status.textContent = "";
    replay.hidden = !speech;
    result.hidden = false;
    say(sentence.textContent, { auto: true });
  } catch (error) {
    if (session !== cameraSession) return;
    console.warn("Recognition failed", error);
    status.textContent = "توقّف التعرّف. اضغط «ماهذا؟» للمتابعة من الملفات المحفوظة.";
  } finally {
    if (session === cameraSession) {
      canvas.width = 0;
      canvas.height = 0;
      action.disabled = false;
      busy = false;
    }
  }
}

action.addEventListener("click", () => stream ? recognize() : openCamera());
stop.addEventListener("click", () => closeCamera());
replay.addEventListener("click", () => { if (spokenWord) say(spokenWord); });
window.addEventListener("pagehide", () => closeCamera());

if (speech && window.SpeechSynthesisUtterance && typeof voiceDialog.showModal === "function") {
  voiceSettings.hidden = false;
  voiceSettings.addEventListener("click", () => {
    draftVoice = selectedVoice;
    refreshVoiceOptions();
    voiceDialog.returnValue = "";
    voiceDialog.showModal();
  });
  speech.addEventListener?.("voiceschanged", () => {
    if (voiceDialog.open) refreshVoiceOptions();
  });
  voiceSelect.addEventListener("change", () => { draftVoice = voiceSelect.value; });
  voicePreview.addEventListener("click", () => {
    refreshVoiceOptions();
    say("Hello! This is a book.", { voiceId: voiceSelect.value, preview: true });
  });
  voiceDialog.addEventListener("close", () => {
    speech.cancel();
    if (voiceDialog.returnValue !== "save") return;
    selectedVoice = draftVoice;
    try { localStorage.setItem(voiceStorageKey, selectedVoice); } catch {}
  });
}
