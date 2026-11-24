
"use strict";

let detector = null;

function toBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  throw new Error("Invalid binary data");
}

async function fetchBytes(url, expectedLength) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());

  if (
    expectedLength !== undefined &&
    bytes.byteLength !== expectedLength
  ) {
    throw new Error(
      `Invalid size for ${url}: expected ${expectedLength}, got ${bytes.byteLength}`
    );
  }

  return bytes;
}

async function loadExternalWeights(baseUrl, manifest) {
  if (
    manifest.source !== "image_encoder.onnx.data" ||
    !Array.isArray(manifest.parts) ||
    !manifest.parts.length
  ) {
    throw new Error("Invalid SigLIP external weights manifest");
  }

  const totalBytes = manifest.total_bytes;

  const weights = new Uint8Array(totalBytes);
  let offset = 0;

  for (let i = 0; i < manifest.parts.length; i++) {
    const part = manifest.parts[i];

    const bytes = await fetchBytes(
      new URL(part.name, baseUrl).href,
      part.bytes
    );

    if (offset + bytes.length > totalBytes) {
      throw new Error("External weights exceed expected size");
    }

    weights.set(bytes, offset);
    offset += bytes.length;

    self.postMessage({
      progress: {
        stage: "weights",
        loaded: i + 1,
        total: manifest.parts.length,
        bytesLoaded: offset,
        bytesTotal: totalBytes
      }
    });
  }

  if (offset !== totalBytes) {
    throw new Error(
      `Incomplete external weights: expected ${totalBytes}, got ${offset}`
    );
  }

  // Verify the complete file against the original SHA-256.
  const digest = await crypto.subtle.digest("SHA-256", weights);

  const hash = Array.from(new Uint8Array(digest))
    .map(value => value.toString(16).padStart(2, "0"))
    .join("");

  if (hash !== manifest.sha256) {
    throw new Error("SigLIP external weights SHA-256 mismatch");
  }

  return weights;
}


async function loadRuntime(runtimeScript, runtimeModule, wasm) {
  const scriptUrl = URL.createObjectURL(
    new Blob([runtimeScript], {
      type: "text/javascript"
    })
  );

  // Use the actual HTTP URL for the ES module.
  const moduleUrl = new URL(
  "vendor/expanded-lite/ort-wasm-simd-threaded.mjs?v=2",
  self.location.href
).href;

  try {
    importScripts(scriptUrl);

    ort.env.wasm.numThreads = 1;

    ort.env.wasm.wasmPaths = {
      mjs: moduleUrl
    };

    ort.env.wasm.wasmBinary = wasm.buffer;

  } finally {
    URL.revokeObjectURL(scriptUrl);
  }
}


async function prepareSiglip(data) {
  const {
    model,
    wasm,
    runtimeScript,
    runtimeModule,
    embeddings,
    labels,
    metadata,
    weightsBaseUrl,
    manifest
  } = data;

  self.postMessage({
    progress: {
      stage: "runtime"
    }
  });

  const wasmBytes = toBytes(wasm);
  const modelBytes = toBytes(model);
  const embeddingBytes = toBytes(embeddings);
  const labelsBytes = toBytes(labels);
  const metadataBytes = toBytes(metadata);

  const manifestData =
  manifest instanceof ArrayBuffer ||
  ArrayBuffer.isView(manifest)
    ? JSON.parse(
        new TextDecoder().decode(manifest)
      )
    : manifest;

  const metadataObject = JSON.parse(
    new TextDecoder().decode(metadataBytes)
  );

  const vocabulary = JSON.parse(
    new TextDecoder().decode(labelsBytes)
  );

  const dimensions = metadataObject.embedding_dimension;
  const count = metadataObject.number_of_words;

  if (
    metadataObject.model !== "google/siglip-base-patch16-224" ||
    dimensions !== 768 ||
    count !== 1000 ||
    metadataObject.dtype !== "float32" ||
    metadataObject.byte_order !== "little-endian" ||
    metadataObject.format !== "raw-binary"
  ) {
    throw new Error("Invalid SigLIP browser metadata");
  }

  if (
    !Array.isArray(vocabulary) ||
    vocabulary.length !== count
  ) {
    throw new Error("Invalid SigLIP labels");
  }

  if (embeddingBytes.byteLength !== count * dimensions * 4) {
    throw new Error("Invalid SigLIP embedding binary size");
  }

  const vectors = new Float32Array(
    embeddingBytes.buffer,
    embeddingBytes.byteOffset,
    count * dimensions
  );

  for (let i = 0; i < vectors.length; i++) {
    if (!Number.isFinite(vectors[i])) {
      throw new Error("Non-finite value in SigLIP embeddings");
    }
  }

  self.postMessage({
    progress: {
      stage: "weights"
    }
  });

  const weights = await loadExternalWeights(
    weightsBaseUrl,
    manifestData
  );

  await loadRuntime(
    runtimeScript,
    runtimeModule,
    wasmBytes
  );

  self.postMessage({
    progress: {
      stage: "session"
    }
  });

  let session;

  try {
    session = await ort.InferenceSession.create(
      modelBytes,
      {
        executionProviders: ["wasm"],
        externalData: [
          {
            path: manifestData.source,
            data: weights
          }
        ]
      }
    );

    ort.env.wasm.wasmBinary = undefined;

    if (!session.inputNames.includes("pixel_values")) {
      throw new Error("SigLIP input pixel_values was not found");
    }

    if (!session.outputNames.includes("image_embeddings")) {
      throw new Error("SigLIP output image_embeddings was not found");
    }

    if (session.inputNames.length !== 1) {
      throw new Error("Unexpected SigLIP input count");
    }

    if (session.outputNames.length !== 1) {
      throw new Error("Unexpected SigLIP output count");
    }

    // Verify that the model can execute.
    const dummyInput = new ort.Tensor(
      "float32",
      new Float32Array(3 * 224 * 224),
      [1, 3, 224, 224]
    );

    let dummyOutput;

    try {
      const outputs = await session.run({
        pixel_values: dummyInput
      });

      dummyOutput = outputs.image_embeddings;

      if (
        !dummyOutput ||
        dummyOutput.data.length !== dimensions
      ) {
        throw new Error("Invalid SigLIP output dimensions");
      }

      let normSquared = 0;

      for (let i = 0; i < dimensions; i++) {
        normSquared += dummyOutput.data[i] ** 2;
      }

      const norm = Math.sqrt(normSquared);

      if (
        !Number.isFinite(norm) ||
        Math.abs(norm - 1) > 0.01
      ) {
        throw new Error(
          `SigLIP output is not normalized: norm=${norm}`
        );
      }

    } finally {
      dummyInput.dispose();
      dummyOutput?.dispose();
    }

    return {
      session,
      labels: vocabulary,
      vectors,
      dimensions
    };

  } catch (error) {
    session?.release();
    throw error;
  }
}

// SigLIP image preprocessing.
async function preprocessImage(pixels, width, height) {
  const sourceCanvas = new OffscreenCanvas(width, height);
  const sourceContext = sourceCanvas.getContext("2d");

  sourceContext.putImageData(
    new ImageData(
      new Uint8ClampedArray(pixels),
      width,
      height
    ),
    0,
    0
  );

  const canvas = new OffscreenCanvas(224, 224);
  const context = canvas.getContext("2d");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  // Resize directly to the processor's 224x224 size.
  context.drawImage(
    sourceCanvas,
    0,
    0,
    224,
    224
  );

  const rgba = context.getImageData(
    0,
    0,
    224,
    224
  ).data;

  const size = 224 * 224;
  const input = new Float32Array(size * 3);

  for (let i = 0; i < size; i++) {
    input[i] = rgba[i * 4] / 127.5 - 1;
    input[i + size] = rgba[i * 4 + 1] / 127.5 - 1;
    input[i + size * 2] = rgba[i * 4 + 2] / 127.5 - 1;
  }

  return input;
}

async function recognizeSiglip(pixels, width, height) {
  if (!detector) {
    throw new Error("SigLIP model is not initialized");
  }

  const {
    session,
    labels,
    vectors,
    dimensions
  } = detector;

  const inputData = await preprocessImage(
    pixels,
    width,
    height
  );

  const input = new ort.Tensor(
    "float32",
    inputData,
    [1, 3, 224, 224]
  );

  let output;

  try {
    const outputs = await session.run({
      pixel_values: input
    });

    output = outputs.image_embeddings;

    const embedding = output.data;

    if (embedding.length !== dimensions) {
      throw new Error("Unexpected SigLIP embedding size");
    }

    const top = [];

    for (let i = 0; i < labels.length; i++) {
      const base = i * dimensions;
      let score = 0;

      for (let j = 0; j < dimensions; j++) {
        score += vectors[base + j] * embedding[j];
      }

      if (
        top.length < 3 ||
        score > top[top.length - 1].score
      ) {
        top.push({
          word: labels[i],
          score
        });

        top.sort((a, b) => b.score - a.score);

        if (top.length > 3) {
          top.pop();
        }
      }
    }

    return {
      model: "SigLIP",
      top: top.map(item => ({
        word: item.word,
        score: Math.round(item.score * 10000) / 10000
      }))
    };

  } finally {
    input.dispose();
    output?.dispose();
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "prepare") {
      detector = await prepareSiglip(data);

      self.postMessage({
        ready: true,
        model: "SigLIP",
        count: detector.labels.length,
        dimensions: detector.dimensions
      });

      return;
    }

    if (data.type === "test") {
      if (!detector) {
        throw new Error("SigLIP model is not initialized");
      }

      self.postMessage({
        ready: true,
        message: "SigLIP model is loaded and inference test passed"
      });

      return;
    }

    if (data.type === "recognize") {
      const result = await recognizeSiglip(
        data.pixels,
        data.width ?? 256,
        data.height ?? 256
      );

      self.postMessage(result);
      return;
    }

    throw new Error(`Unknown message type: ${data.type}`);

  } catch (error) {
    self.postMessage({
      error: error?.message || String(error)
    });
  }
};
