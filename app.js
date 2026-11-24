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
const historyButton = document.getElementById("history-button");
const historyDialog = document.getElementById("history-dialog");
const historyClose = document.getElementById("history-close");
const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const historyClear = document.getElementById("history-clear");
const voiceSelect = document.getElementById("voice-select");
const voicePreview = document.getElementById("voice-preview");
const voiceNote = document.getElementById("voice-note");
const inputModes = document.getElementById("input-modes");
const manualMode = document.getElementById("manual-mode");
const micMode = document.getElementById("mic-mode");
const micPanel = document.getElementById("mic-panel");
const modelPanel = document.getElementById("model-panel");
const modelSelect = document.getElementById("model-select");

let selectedModel = "mobileclip";

modelSelect.addEventListener("change", () => {
  selectedModel = modelSelect.value;
  console.log("Selected model:", selectedModel);
});
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const speech = window.speechSynthesis;
const voiceStorageKey = "misk-english-voice";
const historyStorageKey = "misk-recognition-history";
const maxHistoryItems = 30;
let selectedVoice = "";
let draftVoice = "";
try { selectedVoice = localStorage.getItem(voiceStorageKey) ?? ""; } catch {}

let stream;
let cameraSession = 0;
let busy = false;
let spokenWord = "";
let expandedWorker;
let cancelExpanded;
let clipWorker;
let cancelClip;
let siglipWorker;
let cancelSiglip;
let cameraReady = false;
let micEnabled = false;
let spokenUtterance;
let speechTimer;

// Child age group
const ageScreen = document.getElementById("age-screen");
const ageSelect = document.getElementById("age-select");
const ageStart = document.getElementById("age-start");
const ageError = document.getElementById("age-error");

const ageStorageKey = "misk-child-age-group";
const validAgeGroups = ["4-6", "7-9", "10-11"];

// Show age selection only when no valid age is saved
let savedAgeGroup = "";

try {
  savedAgeGroup = localStorage.getItem(ageStorageKey) || "";
} catch (error) {
  console.warn("Could not read saved age group", error);
}

ageScreen.hidden = validAgeGroups.includes(savedAgeGroup);

ageStart.addEventListener("click", () => {
  const selectedAge = ageSelect.value;

  if (!validAgeGroups.includes(selectedAge)) {
    ageError.hidden = false;
    return;
  }

  try {
    localStorage.setItem(ageStorageKey, selectedAge);
    savedAgeGroup = selectedAge;
    ageError.hidden = true;
    ageScreen.hidden = true;
  } catch (error) {
    console.warn("Could not save age group", error);
    ageError.textContent = "تعذّر حفظ العمر. حاول مرة أخرى.";
    ageError.hidden = false;
  }
});

ageSelect.addEventListener("change", () => {
  ageError.hidden = true;
});

const speechInput = createSpeechInput({
  canListen: () =>
  cameraReady &&
  !busy &&
  !spokenUtterance &&
  !voiceDialog.open &&
  !historyDialog.open &&
  !document.hidden,
  onQuestion: () => {
    if (!busy && camera.readyState >= 2 && camera.videoWidth) recognize("voice");
    else speechInput.refresh();
  },
  onError: (message) => {
    setInputMode(false);
    status.textContent = message;
  },
});
const expandedFolder = new URL("vendor/expanded-lite/", document.baseURI);
const expandedCacheName = "misk-expanded-model-v5";
const clipFolder = new URL("vendor/clip-vit-base-patch32/",document.baseURI);
const clipIndexFolder = new URL("index/", clipFolder);
const clipCacheName = "misk-clip-vit-base-patch32-v1";
const clipModelBytes = 88648877;
// SigLIP Base assets
const siglipFolder = new URL("vendor/siglip/", document.baseURI);
const siglipOnnxFolder = new URL("onnx/", siglipFolder);
const siglipPartsFolder = new URL("external_parts/", siglipOnnxFolder);
const siglipIndexFolder = new URL("index/", siglipFolder);
const siglipCacheName = "misk-siglip-base-v1";

// GitHub Pages lets browsers keep the page for a while, so phones can keep
// running an old version. Compare with version.json and reload when it changed.
const appVersion = "15";
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
  speechInput.refresh();
  if (document.visibilityState === "visible") checkForUpdate();
});

function loadHistory() {
  try {
    const history = JSON.parse(
      localStorage.getItem(historyStorageKey) || "[]"
    );

    if (!Array.isArray(history)) return [];

    // Remove Face records to protect privacy.
    const safeHistory = history.filter(
      (record) =>
        record &&
        String(record.word || "").trim().toLowerCase() !== "face"
    );

    // Update stored history if Face records were found.
    if (safeHistory.length !== history.length) {
      localStorage.setItem(
        historyStorageKey,
        JSON.stringify(safeHistory)
      );
    }

    return safeHistory;
  } catch {
    return [];
  }
}

function saveHistory(label, model) {
  if (String(label).trim().toLowerCase() === "face") {
    return false;
  }

  try {
    const image = canvas.toDataURL("image/jpeg", 0.65);

    const record = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      word: label,
      model: model,
      sentence: englishSentence(label),
      image,
      date: new Date().toISOString(),
    };

    const history = loadHistory();
    history.unshift(record);

    while (history.length > maxHistoryItems) {
      history.pop();
    }

    localStorage.setItem(historyStorageKey, JSON.stringify(history));
    return true;
  } catch (error) {
    console.warn("Could not save history", error);
    return false;
  }
}

function renderHistory() {
  const history = loadHistory();

  historyList.replaceChildren();
  historyEmpty.hidden = history.length > 0;

  for (const record of history) {
    const item = document.createElement("div");
    item.className = "history-item";

    const image = document.createElement("img");
    image.src = record.image;
    image.alt = record.word;

    const info = document.createElement("div");
    info.className = "history-item-info";

    const word = document.createElement("span");
    word.className = "history-item-word";
    word.textContent = record.word;

    const model = document.createElement("span");
    model.className = "history-item-model";

    const modelNames = {
      mobileclip: "MobileCLIP-S0",
      clip: "CLIP ViT-B/32",
      siglip: "SigLIP Base"
    };

    model.textContent = modelNames[record.model] || "غير محدد";

    const date = document.createElement("span");
    date.className = "history-item-date";

    const parsedDate = new Date(record.date);
    date.textContent = Number.isNaN(parsedDate.getTime())
      ? ""
      : parsedDate.toLocaleString();

    info.append(word, model, date);
    item.append(image, info);
    historyList.append(item);
  }
}

function setInputMode(useMic) {
  micEnabled = useMic;
  manualMode.setAttribute("aria-pressed", String(!useMic));
  micMode.setAttribute("aria-pressed", String(useMic));
  micPanel.hidden = !useMic;
  action.hidden = useMic;
  if (useMic) speechInput.enable();
  else speechInput.disable();
}

function cancelSpeech() {
  clearTimeout(speechTimer);
  spokenUtterance = undefined;
  speech?.cancel();
}

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


async function downloadClip(session) {
  const controller = new AbortController();
  const { signal } = controller;

  cancelClip = () => controller.abort();

  const cache = await window.caches
    ?.open(clipCacheName)
    .catch(() => null);

  const progress = new Map();
  let total = 0;
  let retrying = 0;
  let lastMessage;

  function report() {
    if (session !== cameraSession) return;

    const sum = [...progress.values()]
      .reduce((a, b) => a + b, 0);

    const percent = total
      ? Math.min(99, Math.floor(100 * sum / total))
      : 0;

    const caption = retrying
      ? "جاري إعادة محاولة الاتصال"
      : "جاري تحميل نموذج CLIP";

    const message = `${caption}… ${percent}٪`;

    if (message !== lastMessage) {
      lastMessage = message;
      status.textContent = message;
    }
  }

  async function part(name, folder, expectedBytes) {
    const url = new URL(name, folder).href;

    const cached = await cache?.match(url).catch(() => null);

    let data = await cached?.arrayBuffer().catch(() => null);

    if (
      !data ||
      (expectedBytes && data.byteLength !== expectedBytes)
    ) {
      let lastError;

      for (
        let attempt = 0;
        attempt < 3 && !signal.aborted;
        attempt++
      ) {
        if (attempt) {
          retrying++;
          report();
        }

        try {
          const requestUrl = new URL(url);

          if (attempt) {
            requestUrl.searchParams.set(
              "retry",
              String(Date.now())
            );
          }

          data = await downloadExpandedPart(
            requestUrl.href,
            signal,
            (loaded) => {
              if (expectedBytes) {
                progress.set(name, loaded);
              }
              report();
            }
          );

          if (
            expectedBytes &&
            data.byteLength !== expectedBytes
          ) {
            throw new Error("Incomplete CLIP file");
          }

          if (expectedBytes) {
            await cache?.put(
              url,
              new Response(data)
            ).catch(() => {});
          }

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

      if (!data || signal.aborted) {
        throw lastError ?? new Error("CLIP download canceled");
      }
    }

    if (expectedBytes) {
      progress.set(name, data.byteLength);
      report();
    }

    return new Uint8Array(data);
  }

  async function assemble(entry, folder) {
    const buffer = new Uint8Array(entry.bytes);
    let offset = 0;

    for (const { name, bytes } of entry.parts) {
      const piece = await part(name, folder, bytes);

      buffer.set(piece, offset);
      offset += piece.length;
    }

    if (offset !== entry.bytes) {
      throw new Error("Incomplete CLIP asset");
    }

    return buffer;
  }

  try {
    report();

    // Read CLIP vocabulary index manifest.
    const clipManifestBytes = await part(
      "assets.json",
      clipIndexFolder
    );

    const clipAssets = JSON.parse(
      new TextDecoder().decode(clipManifestBytes)
    );

    // Read the existing ONNX Runtime manifest.
    const runtimeManifestBytes = await part(
      "assets.json",
      expandedFolder
    );

    const runtimeAssets = JSON.parse(
      new TextDecoder().decode(runtimeManifestBytes)
    );

    // Calculate total download size.
    total =
      clipModelBytes +
      clipAssets.index.bytes +
      clipAssets.labels.bytes +
      runtimeAssets.runtime.bytes +
      runtimeAssets.script.bytes +
      runtimeAssets.module.bytes;

    // Download the model, index, vocabulary and runtime.
    const [
      model,
      index,
      vocabulary,
      wasm,
      runtimeScript,
      runtimeModule
    ] = await Promise.all([
      part(
        "vision_model_int8.onnx",
        clipFolder,
        clipModelBytes
      ),

      assemble(clipAssets.index, clipIndexFolder),

      assemble(clipAssets.labels, clipIndexFolder),

      assemble(runtimeAssets.runtime, expandedFolder),

      assemble(runtimeAssets.script, expandedFolder),

      assemble(runtimeAssets.module, expandedFolder)
    ]);

    if (session !== cameraSession) {
      throw new Error("Camera session ended");
    }

    // Expected sizes after decompression inside clip-worker.js.
    const lengths = {
      model: clipModelBytes,
      wasm: runtimeAssets.runtime.expandedBytes,
      vocabulary: clipAssets.labels.expandedBytes,
      index: clipAssets.index.expandedBytes
    };

    return {
      model,
      index,
      vocabulary,
      wasm,
      runtimeScript,
      runtimeModule,
      lengths
    };

  } finally {
    controller.abort();

    if (session === cameraSession) {
      cancelClip = undefined;
    }
  }
}


async function downloadSiglip(session) {
  const controller = new AbortController();
  const { signal } = controller;

  cancelSiglip = () => controller.abort();

  const cache = await window.caches
    ?.open(siglipCacheName)
    .catch(() => null);

  const progress = new Map();
  let total = 0;
  let retrying = 0;
  let lastMessage = "";

  function report() {
    if (session !== cameraSession) return;

    const downloaded = [...progress.values()]
      .reduce((sum, value) => sum + value, 0);

    const percent = total
      ? Math.min(99, Math.floor(downloaded / total * 100))
      : 0;

    const caption = retrying
      ? "جاري إعادة محاولة الاتصال"
      : "جاري تحميل نموذج SigLIP";

    const message = `${caption}… ${percent}٪`;

    if (message !== lastMessage) {
      lastMessage = message;
      status.textContent = message;
    }
  }

  async function part(url, expectedBytes, cacheResult = true) {
    const cached = cacheResult
      ? await cache?.match(url).catch(() => null)
      : null;

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

          if (attempt) {
            requestUrl.searchParams.set("retry", String(Date.now()));
          }

          data = await downloadExpandedPart(
            requestUrl.href,
            signal,
            (loaded) => {
              if (expectedBytes) progress.set(url, loaded);
              report();
            }
          );

          if (expectedBytes && data.byteLength !== expectedBytes) {
            throw new Error("Incomplete SigLIP asset");
          }

          if (cacheResult && expectedBytes) {
            await cache?.put(url, new Response(data)).catch(() => {});
          }

          lastError = null;
          break;
        } catch (error) {
          data = null;
          progress.delete(url);
          lastError = error;
        } finally {
          if (attempt) retrying--;
          report();
        }
      }

      if (!data || signal.aborted) {
        throw lastError ?? new Error("SigLIP download canceled");
      }
    }

    if (expectedBytes) {
      progress.set(url, data.byteLength);
      report();
    }

    return new Uint8Array(data);
  }

  async function assemble(entry, folder) {
    const buffer = new Uint8Array(entry.bytes);
    let offset = 0;

    for (const item of entry.parts) {
      const url = new URL(item.name, folder).href;
      const piece = await part(url, item.bytes);

      buffer.set(piece, offset);
      offset += piece.byteLength;
    }

    if (offset !== entry.bytes) {
      throw new Error("Incomplete SigLIP runtime");
    }

    return buffer;
  }

  try {
    report();

    // Read the shared ONNX Runtime manifest.
    const runtimeManifestUrl = new URL(
      "assets.json",
      expandedFolder
    ).href;

    const runtimeManifestBytes = await part(
      runtimeManifestUrl,
      0,
      false
    );

    const runtimeAssets = JSON.parse(
      new TextDecoder().decode(runtimeManifestBytes)
    );

    const runtimeFolder = expandedFolder;

    const modelUrl = new URL(
      "image_encoder.onnx",
      siglipOnnxFolder
    ).href;

    const embeddingsUrl = new URL(
      "embeddings.bin",
      siglipIndexFolder
    ).href;

    const labelsUrl = new URL(
      "labels.json",
      siglipIndexFolder
    ).href;

    const metadataUrl = new URL(
      "web_metadata.json",
      siglipIndexFolder
    ).href;

    const manifestUrl = new URL(
      "manifest.json",
      siglipPartsFolder
    ).href;

    const runtimeScriptUrl = new URL(
      "ort.wasm.min.js",
      runtimeFolder
    ).href;

    const runtimeModuleUrl = new URL(
      "ort-wasm-simd-threaded.mjs",
      runtimeFolder
    ).href;

    const runtimeBytes = runtimeAssets.runtime.bytes;
    const runtimeScriptBytes = 95707;
    const runtimeModuleBytes = 24274;

    total =
      1604365 +
      3072000 +
      14139 +
      192 +
      0 +
      runtimeBytes +
      runtimeScriptBytes +
      runtimeModuleBytes;

    const [
      model,
      embeddings,
      labels,
      metadata,
      manifest,
      runtimeCompressed,
      runtimeScript,
      runtimeModule
    ] = await Promise.all([
      part(modelUrl, 1604365),
      part(embeddingsUrl, 3072000),
      part(labelsUrl, 14139),
      part(metadataUrl, 192),
      part(manifestUrl, 0, false),
      assemble(runtimeAssets.runtime, runtimeFolder),
      part(runtimeScriptUrl, runtimeScriptBytes, false),
      part(runtimeModuleUrl, runtimeModuleBytes, false)
    ]);

    // The six runtime parts form one gzip stream.
    const wasm = await new Response(
      new Blob([runtimeCompressed]).stream().pipeThrough(
        new DecompressionStream("gzip")
      )
    ).arrayBuffer();

    if (wasm.byteLength !== runtimeAssets.runtime.expandedBytes) {
      throw new Error("ONNX Runtime WASM size mismatch");
    }

    if (session !== cameraSession) {
      throw new Error("Camera session ended");
    }

    return {
      model,
      wasm,
      runtimeScript,
      runtimeModule,
      embeddings,
      labels,
      metadata,
      weightsBaseUrl: siglipPartsFolder.href,
      manifest
    };

  } finally {
    controller.abort();

    if (session === cameraSession) {
      cancelSiglip = undefined;
    }
  }
}

function requestExpanded(message, session) {
  expandedWorker ??= new Worker("expanded-worker.js?v=12");
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


function requestClip(message, session) {
  clipWorker ??= new Worker("clip-worker.js?v=1");

  const worker = clipWorker;

  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => fail(new Error("انتهت مهلة تجهيز CLIP.")),
      180000
    );

    const fail = (error) => {
      clearTimeout(timer);
      worker.terminate();

      if (clipWorker === worker) {
        clipWorker = undefined;
      }

      cancelClip = undefined;
      reject(error);
    };

    cancelClip = () => {
      fail(new Error("CLIP operation canceled"));
    };

    worker.onerror = (event) => {
      fail(new Error(event.message || "CLIP worker error"));
    };

    worker.onmessage = ({ data }) => {
      if (data.error) {
        fail(new Error(data.error));
        return;
      }

      clearTimeout(timer);
      cancelClip = undefined;
      resolve(data);
    };

    const buffers = Object.values(message)
      .filter(ArrayBuffer.isView)
      .map((value) => value.buffer);

    try {
      worker.postMessage(message, buffers);
    } catch (error) {
      fail(error);
    }
  });
}


function requestSiglip(message, session) {
  siglipWorker ??= new Worker("siglip-worker.js?v=1");

  const worker = siglipWorker;

  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => fail(new Error("انتهت مهلة تجهيز SigLIP.")),
      600000
    );

    const fail = (error) => {
      clearTimeout(timer);
      worker.terminate();

      if (siglipWorker === worker) {
        siglipWorker = undefined;
      }

      cancelSiglip = undefined;
      reject(error);
    };

    cancelSiglip = () => {
      fail(new Error("SigLIP operation canceled"));
    };

    worker.onerror = (event) => {
      fail(new Error(event.message || "SigLIP worker error"));
    };

    worker.onmessage = ({ data }) => {
  if (data.error) {
    fail(new Error(data.error));
    return;
  }

  // Ignore progress messages and wait for the final result
  if (
    data.progress &&
    !data.ready &&
    !data.model &&
    !data.top &&
    !data.message
  ) {
    return;
  }

  clearTimeout(timer);
  cancelSiglip = undefined;
  resolve(data);
};

    if (session !== cameraSession) {
      fail(new Error("Camera session ended"));
      return;
    }

    const buffers = [
      ...new Set(
        Object.values(message)
          .filter(ArrayBuffer.isView)
          .map((value) => value.buffer)
      )
    ];

    try {
      worker.postMessage(message, buffers);
    } catch (error) {
      fail(error);
    }
  });
}



async function prepareClip(session) {
  if (!clipWorker) {
    const assets = await downloadClip(session);

    if (session !== cameraSession) {
      return false;
    }

    status.textContent = "اكتمل تحميل CLIP. جاري تجهيز النموذج…";

    const response = await requestClip(
      {
        type: "prepare",
        ...assets
      },
      session
    );

  }

  return session === cameraSession;
}


async function prepareSiglip(session) {
  if (!siglipWorker) {
    const assets = await downloadSiglip(session);

    if (session !== cameraSession) {
      return false;
    }

    status.textContent =
      "اكتمل تحميل SigLIP. جاري تجهيز النموذج…";

    const response = await requestSiglip(
      {
        type: "prepare",
        ...assets
      },
      session
    );

  }

  return session === cameraSession;
}


window.testSiglipModel = async function () {
  const session = cameraSession;

  try {
    status.textContent = "جاري اختبار SigLIP…";

    const ready = await prepareSiglip(session);

    if (!ready || session !== cameraSession) {
      return;
    }

    const response = await requestSiglip(
      { type: "test" },
      session
    );

    console.log("SigLIP test result:", response);

    status.textContent = "نجح تحميل واختبار SigLIP.";

    return response;

  } catch (error) {
    console.error("SigLIP test failed:", error);

    status.textContent =
      `فشل اختبار SigLIP: ${error.message}`;
  }
};


async function recognizeClip(session) {
  const pixels = context.getImageData(0, 0, 256, 256).data;

  if (!clipWorker) {
    await prepareClip(session);
  }

  if (session !== cameraSession) {
    return { label: "" };
  }

  status.textContent = "جاري التعرّف باستخدام CLIP…";

  const ageLevels = {
    "4-6": 1,
    "7-9": 2,
    "10-11": 3
  };

  const maxLevel = ageLevels[savedAgeGroup] ?? 3;

  const {
    label,
    status: recognitionStatus = "recognized",
    detectedLevel,
    top = [],
    levelCandidates = []
  } = await requestClip(
    {
      type: "recognize",
      pixels,
      width: 256,
      height: 256,
      maxLevel
    },
    session
  );

  return {
    label,
    status: recognitionStatus,
    detectedLevel,
    top,
    levelCandidates
  };
}

async function recognizeSiglip(session) {
  if (!siglipWorker) {
    await prepareSiglip(session);
  }

  if (session !== cameraSession) {
    return { label: "" };
  }

  const pixels = context.getImageData(
    0,
    0,
    256,
    256
  ).data;

  status.textContent = "جاري التعرّف باستخدام SigLIP…";

  const result = await requestSiglip(
    {
      type: "recognize",
      pixels,
      width: 256,
      height: 256
    },
    session
  );

  if (session !== cameraSession) {
    return { label: "" };
  }

  return result;
}


window.testClipRecognition = async function () {
  if (busy || !stream) {
    console.warn("الكاميرا مشغولة أو غير مفعّلة.");
    return;
  }

  const startedAt = performance.now();
  const session = cameraSession;

  try {
    busy = true;
    action.disabled = true;

    takeSnapshot();

    const recognitionStart = performance.now();

    const {
      label,
      status: recognitionStatus = "recognized",
      detectedLevel,
      top = [],
      levelCandidates = []
    } = await recognizeClip(session);

    if (session !== cameraSession) return;

    const recognitionMs = Math.round(
      performance.now() - recognitionStart
    );

    const totalMs = Math.round(
      performance.now() - startedAt
    );

    logAttempt({
      trigger: "CLIP",
      answer: label,
      recognitionStatus,
      detectedLevel,
      top,
      levelCandidates,
      recognitionMs,
      totalMs
    });

    console.log("CLIP result:", {
      label,
      recognitionStatus,
      detectedLevel,
      top,
      levelCandidates,
      recognitionMs,
      totalMs
    });

    return { label, top, recognitionMs };

  } catch (error) {
    console.error("CLIP test failed:", error);

  } finally {
    if (session === cameraSession) {
      canvas.width = 0;
      canvas.height = 0;
      action.disabled = false;
      busy = false;
    }
  }
};


window.testClipModel = async function () {
  const session = cameraSession;

  try {
    status.textContent = "جاري اختبار CLIP…";

    const ready = await prepareClip(session);

    if (!ready || session !== cameraSession) {
      return;
    }

    const result = await requestClip(
      { type: "test" },
      session
    );

    console.log("CLIP test result:", result);

    status.textContent = "نجح تحميل واختبار CLIP ViT-B/32.";

    return result;

  } catch (error) {
    console.error("CLIP test failed:", error);
    status.textContent = `فشل اختبار CLIP: ${error.message}`;
  }
};


async function recognizeExpanded(session) {
  const pixels = context.getImageData(0, 0, 256, 256).data;
  if (!expandedWorker) {
    const assets = await downloadExpanded(session);
    if (session !== cameraSession) return "";
    status.textContent = "اكتمل التحميل. جاري تجهيز التعرّف…";
    await requestExpanded({ type: "prepare", ...assets }, session);
  }
  if (session !== cameraSession) return { label: "" };
  status.textContent = "جاري التعرّف…";
  const ageLevels = {
  "4-6": 1,
  "7-9": 2,
  "10-11": 3
};

const maxLevel = ageLevels[savedAgeGroup] ?? 3;

const {
  label,
  status: recognitionStatus = "recognized",
  detectedLevel,
  top = [],
  levelCandidates = []
} = await requestExpanded({ type: "recognize", pixels, maxLevel }, session);

return {
  label,
  status: recognitionStatus,
  detectedLevel,
  top,
  levelCandidates
};
}

// Measurement for testing (?debug=1): one record per question, kept in
// window.showSayLog and printed to the console. No image is stored.
const debugMode = new URLSearchParams(location.search).has("debug");
window.showSayLog = [];
function logAttempt(record) {
  if (!debugMode) return;

  window.showSayLog.push(record);

  const guesses = record.top
    .map(({ word, score }) => `${word} ${score}`)
    .join(" | ");

  const levels = record.levelCandidates
    .map(({ word, level, score }) => `Level ${level}: ${word} ${score}`)
    .join(" | ");

  console.info(
    `[measure] #${window.showSayLog.length} ${record.trigger}: `
    + `answer="${record.answer || "(not sure)"}" `
    + `status="${record.recognitionStatus}" `
    + `detectedLevel="${record.detectedLevel ?? "-"}" `
    + `top3: ${guesses} | `
    + `best pre level: ${levels} | `
    + `recognition ${record.recognitionMs} ms, `
    + `total ${record.totalMs} ms`
  );
}
function copyMeasurements() {
  const header = "n,trigger,answer,top1,score1,top2,score2,top3,score3,recognition_ms,total_ms";
  const rows = window.showSayLog.map((r, i) => [
    i + 1, r.trigger, r.answer,
    ...[0, 1, 2].flatMap((k) => [r.top[k]?.word ?? "", r.top[k]?.score ?? ""]),
    r.recognitionMs, r.totalMs,
  ].join(","));
  return [header, ...rows].join("\n");
}
window.copyMeasurements = copyMeasurements;

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
  cancelSpeech();
  const utterance = new SpeechSynthesisUtterance(text);
  const voices = availableEnglishVoices();
  utterance.voice = voices.find((voice) => voice.voiceURI === voiceId)
    ?? voices.find((voice) => voice.localService && voice.lang.toLowerCase() === "en-us")
    ?? voices.find((voice) => voice.localService)
    ?? voices.find((voice) => voice.lang.toLowerCase() === "en-us")
    ?? null;
  utterance.lang = utterance.voice?.lang ?? "en-US";
  utterance.rate = 0.85;
  spokenUtterance = utterance;
  speechInput.refresh();
  function finish() {
    if (spokenUtterance !== utterance) return false;
    clearTimeout(speechTimer);
    spokenUtterance = undefined;
    speechInput.refresh();
    return true;
  }
  utterance.onend = finish;
  utterance.onerror = (event) => {
    if (!finish() || event.error === "canceled" || event.error === "interrupted") return;
    if (preview) {
      if (!voiceDialog.open) return;
      voiceNote.hidden = false;
      voiceNote.textContent = "تعذّر تشغيل هذا الصوت. جرّب صوتًا آخر.";
    } else if (auto) status.textContent = "اضغط زر الصوت لسماع الكلمة.";
    else status.textContent = "تعذّر تشغيل الصوت. الكلمة ظاهرة على الشاشة.";
  };
  speechTimer = setTimeout(() => {
    if (spokenUtterance !== utterance) return;
    cancelSpeech();
    speechInput.refresh();
  }, 15000);
  try { speech.speak(utterance); } catch { finish(); }
}

function closeCamera(message = "شغّل الكاميرا ووجّهها نحو غرض واحد.") {
  cameraSession += 1;
  cameraReady = false;
  setInputMode(false);
  inputModes.hidden = true;
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
  cancelSpeech();
  status.textContent = message;
  action.textContent = "شغّل الكاميرا";
  modelPanel.hidden = true;
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
  cameraReady = true;
  inputModes.hidden = false;
  stop.hidden = false;
  action.textContent = "ماهذا؟";
  modelPanel.hidden = false;
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

async function recognize(trigger = "button") {
  if (busy || !stream) return;
  const startedAt = performance.now();
  if (!camera.videoWidth || camera.readyState < 2) {
    status.textContent = "انتظر حتى تظهر الكاميرا وجرّب مرة ثانية.";
    return;
  }
  const session = cameraSession;
  busy = true;
  speechInput.refresh();
  action.disabled = true;
  result.hidden = true;
  spokenWord = "";
  cancelSpeech();
  status.textContent = "جاري التعرّف…";

  try {
    takeSnapshot();
    const recognitionStart = performance.now();
    let recognitionResult;

    if (selectedModel === "clip") {
      recognitionResult = await recognizeClip(session);
    } else if (selectedModel === "siglip") {
      recognitionResult = await recognizeSiglip(session);
    } else {
      recognitionResult = await recognizeExpanded(session);
    }


const {
  label: returnedLabel,
  status: recognitionStatus = "recognized",
  detectedLevel,
  top = [],
  levelCandidates = []
} = recognitionResult;

const label =
  returnedLabel ||
  top?.[0]?.word ||
  "";
    if (session !== cameraSession) return;
    const attemptResult = {
  model: selectedModel,
  answer: label,
  status: recognitionStatus,
  recognitionMs: Math.round(performance.now() - recognitionStart),
  totalMs: Math.round(performance.now() - startedAt),
  top: JSON.stringify(top)
};

logAttempt({
  trigger: `${trigger}/${selectedModel}`,
  answer: label,
  recognitionStatus,
  detectedLevel,
  top,
  levelCandidates,
  recognitionMs: attemptResult.recognitionMs,
  totalMs: attemptResult.totalMs,
});

console.table([attemptResult]);
    if (recognitionStatus === "higher_level") {
  status.textContent =
    `هذا الغرض في مستوى تعليمي اعلى, واصل التعلم الى ان تصل اليه`;
  return;
}

if (recognitionStatus === "unknown" || !label) {
  status.textContent =
    "لم أتمكن من التعرّف على الغرض. قرّبه داخل الإطار وجرّب مرة ثانية.";
  return;
}
    saveHistory(label, selectedModel);
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
    status.textContent = micEnabled
      ? "توقّف التعرّف. اسأل مرة ثانية للمتابعة من الملفات المحفوظة."
      : "توقّف التعرّف. اضغط «ماهذا؟» للمتابعة من الملفات المحفوظة.";
  } finally {
    if (session === cameraSession) {
      canvas.width = 0;
      canvas.height = 0;
      action.disabled = false;
      busy = false;
      speechInput.refresh();
    }
  }
}

action.addEventListener("click", () => stream ? recognize() : openCamera());
historyButton.addEventListener("click", () => {
  renderHistory();
  historyDialog.showModal();
  speechInput.refresh();
});

historyClose.addEventListener("click", () => {
  historyDialog.close();
});

historyClear.addEventListener("click", () => {
  const confirmed = confirm("هل أنت متأكد من مسح جميع عمليات التعرّف المحفوظة؟");

  if (!confirmed) return;

  try {
    localStorage.removeItem(historyStorageKey);
    renderHistory();
  } catch (error) {
    console.warn("Could not clear history", error);
    alert("تعذّر مسح السجل. حاول مرة أخرى.");
  }
});

historyDialog.addEventListener("close", () => {
  speechInput.refresh();
});
manualMode.addEventListener("click", () => {
  setInputMode(false);
  if (!busy) status.textContent = "حط الغرض داخل الإطار واضغط الزر.";
});
micMode.addEventListener("click", () => {
  if (micEnabled || !cameraReady) return;
  if (!busy) status.textContent = "";
  setInputMode(true);
});
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
    speechInput.refresh();
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
    cancelSpeech();
    speechInput.refresh();
    if (voiceDialog.returnValue !== "save") return;
    selectedVoice = draftVoice;
    try { localStorage.setItem(voiceStorageKey, selectedVoice); } catch {}
  });
}
