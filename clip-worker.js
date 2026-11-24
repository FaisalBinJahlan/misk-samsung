"use strict";
let detector;

function toBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  throw new Error("Invalid binary data");
}

// فك الضغط فقط إذا كانت البيانات مضغوطة بصيغة GZIP.
async function decompressIfGzip(data, expectedLength) {
  const bytes = toBytes(data);

  const isGzip =
    bytes.length >= 2 &&
    bytes[0] === 0x1f &&
    bytes[1] === 0x8b;

  let result;

  if (isGzip) {
    const stream = new Blob([bytes])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));

    result = new Uint8Array(
      await new Response(stream).arrayBuffer()
    );
  } else {
    result = bytes;
  }

  if (
    expectedLength !== undefined &&
    result.byteLength !== expectedLength
  ) {
    throw new Error(
      `Invalid file size: expected ${expectedLength}, got ${result.byteLength}`
    );
  }

  return result;
}

async function prepareDetector(data) {
  const {
    model,
    wasm,
    vocabulary,
    index,
    runtimeScript,
    runtimeModule,
    lengths
  } = data;

  // CLIP vision model is currently downloaded as a raw ONNX file.
  // The other three binary assets are GZIP-compressed.
  const [
    modelBytes,
    wasmBytes,
    labelsBytes,
    indexBytes
  ] = await Promise.all([
    decompressIfGzip(model, lengths?.model),
    decompressIfGzip(wasm, lengths?.wasm),
    decompressIfGzip(vocabulary, lengths?.vocabulary),
    decompressIfGzip(index, lengths?.index)
  ]);

  // Read vocabulary metadata.
  const metadata = JSON.parse(
    new TextDecoder().decode(labelsBytes)
  );

  const {
    labels,
    levels,
    dimensions,
    format
  } = metadata;

  if (
    format !== "int8" ||
    !Array.isArray(labels) ||
    !labels.length ||
    dimensions !== 512
  ) {
    throw new Error("Invalid CLIP vocabulary metadata");
  }

  const vectors = new Int8Array(
    indexBytes.buffer,
    indexBytes.byteOffset,
    indexBytes.byteLength
  );

  if (vectors.length !== labels.length * dimensions) {
    throw new Error("CLIP index dimensions do not match labels");
  }

  // Calculate the inverse norm for every text embedding.
  const inverseNorms = new Float32Array(labels.length);

  for (let i = 0; i < labels.length; i++) {
    let squaredNorm = 0;
    const start = i * dimensions;

    for (let j = 0; j < dimensions; j++) {
      const value = vectors[start + j];
      squaredNorm += value * value;
    }

    inverseNorms[i] =
      squaredNorm > 0 ? 1 / Math.sqrt(squaredNorm) : 0;
  }

  // Load ONNX Runtime.
  const scriptUrl = URL.createObjectURL(
    new Blob([runtimeScript], {
      type: "text/javascript"
    })
  );

  const moduleUrl = URL.createObjectURL(
    new Blob([runtimeModule], {
      type: "text/javascript"
    })
  );

  let session;

  try {
    importScripts(scriptUrl);

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      mjs: moduleUrl
    };
    ort.env.wasm.wasmBinary = wasmBytes.buffer;

    session = await ort.InferenceSession.create(
      modelBytes,
      {
        executionProviders: ["wasm"]
      }
    );

    ort.env.wasm.wasmBinary = undefined;

    // Verify model input and output.
    if (!session.inputNames.includes("pixel_values")) {
      throw new Error("CLIP input pixel_values was not found");
    }

    if (!session.outputNames.includes("image_embeds")) {
      throw new Error("CLIP output image_embeds was not found");
    }

    // Test inference using a dummy 224x224 image.
    const dummyData = new Float32Array(
      3 * 224 * 224
    );

    const dummyInput = new ort.Tensor(
      "float32",
      dummyData,
      [1, 3, 224, 224]
    );

    let dummyOutput;

    try {
      const outputs = await session.run({
        pixel_values: dummyInput
      });

      dummyOutput = outputs.image_embeds;

      if (
        !dummyOutput ||
        dummyOutput.data.length !== 512
      ) {
        throw new Error(
          "Unexpected CLIP embedding dimensions"
        );
      }
    } finally {
      dummyInput.dispose();
      dummyOutput?.dispose();
    }

    return {
      session,
      labels,
      levels: levels ?? labels.map(() => 1),
      dimensions,
      vectors,
      inverseNorms
    };

  } catch (error) {
    session?.release();
    throw error;

  } finally {
    URL.revokeObjectURL(scriptUrl);
    URL.revokeObjectURL(moduleUrl);
  }
}


async function recognizeClip(pixels, width = 256, height = 256, maxLevel = 3) {
  if (!detector) {
    throw new Error("CLIP model is not initialized");
  }

  const {
    session,
    labels,
    levels,
    dimensions,
    vectors,
    inverseNorms
  } = detector;

  // CLIP preprocessing: resize shortest edge to 224,
  // then center crop to 224x224.
  const scale = 224 / Math.min(width, height);

  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);

  const sourceCanvas = new OffscreenCanvas(width, height);
  const sourceContext = sourceCanvas.getContext("2d");

  const imageData = new ImageData(
    new Uint8ClampedArray(pixels),
    width,
    height
  );

  sourceContext.putImageData(imageData, 0, 0);

  const resizedCanvas = new OffscreenCanvas(
    resizedWidth,
    resizedHeight
  );

  const resizedContext = resizedCanvas.getContext("2d");
  resizedContext.imageSmoothingEnabled = true;
  resizedContext.imageSmoothingQuality = "high";

  resizedContext.drawImage(
    sourceCanvas,
    0,
    0,
    resizedWidth,
    resizedHeight
  );

  const cropX = Math.floor((resizedWidth - 224) / 2);
  const cropY = Math.floor((resizedHeight - 224) / 2);

  const finalCanvas = new OffscreenCanvas(224, 224);
  const finalContext = finalCanvas.getContext("2d");

  finalContext.drawImage(
    resizedCanvas,
    cropX,
    cropY,
    224,
    224,
    0,
    0,
    224,
    224
  );

  const finalPixels = finalContext.getImageData(
    0,
    0,
    224,
    224
  ).data;

  // CLIP normalization.
  const mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ];

  const std = [
    0.26862954,
    0.26130258,
    0.27577711
  ];

  const imageSize = 224 * 224;
  const inputData = new Float32Array(imageSize * 3);

  for (let i = 0; i < imageSize; i++) {
    inputData[i] =
      (finalPixels[i * 4] / 255 - mean[0]) / std[0];

    inputData[i + imageSize] =
      (finalPixels[i * 4 + 1] / 255 - mean[1]) / std[1];

    inputData[i + imageSize * 2] =
      (finalPixels[i * 4 + 2] / 255 - mean[2]) / std[2];
  }

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

    output = outputs.image_embeds;

    const embedding = output.data;

    if (embedding.length !== dimensions) {
      throw new Error("Unexpected CLIP embedding size");
    }

    // Normalize image embedding.
    let squaredNorm = 0;

    for (let i = 0; i < dimensions; i++) {
      squaredNorm += embedding[i] * embedding[i];
    }

    const imageNorm = Math.sqrt(squaredNorm);

    if (!imageNorm) {
      throw new Error("Invalid image embedding");
    }

    // Calculate cosine similarity against all vocabulary items.
    const top = [];
    const levelBest = new Map();

    for (let i = 0; i < labels.length; i++) {
      const base = i * dimensions;
      let similarity = 0;

      for (let j = 0; j < dimensions; j++) {
        similarity += vectors[base + j] * embedding[j];
      }

      const score =
        similarity * inverseNorms[i] / imageNorm;

      const level = levels[i];
      const current = levelBest.get(level);

      if (!current || score > current.score) {
        levelBest.set(level, {
          index: i,
          score
        });
      }

      if (
        top.length < 3 ||
        score > top[top.length - 1].score
      ) {
        top.push({
          word: labels[i],
          level,
          score
        });

        top.sort((a, b) => b.score - a.score);

        if (top.length > 3) {
          top.pop();
        }
      }
    }

    const candidates = top.map(item => ({
      word: item.word,
      level: item.level,
      score: Math.round(item.score * 1000) / 1000
    }));

    const levelCandidates = [...levelBest.entries()]
      .map(([level, candidate]) => ({
        word: labels[candidate.index],
        level,
        score: Math.round(candidate.score * 1000) / 1000
      }))
      .sort((a, b) => a.level - b.level);

    const allowed = [...levelBest.entries()]
      .filter(([level]) => level <= maxLevel)
      .map(([level, candidate]) => ({
        index: candidate.index,
        level,
        score: candidate.score
      }));

    const higher = [...levelBest.entries()]
      .filter(([level]) => level > maxLevel)
      .map(([level, candidate]) => ({
        index: candidate.index,
        level,
        score: candidate.score
      }));

    const bestAllowed = allowed.reduce(
      (best, item) =>
        !best || item.score > best.score ? item : best,
      null
    );

    const bestHigher = higher.reduce(
      (best, item) =>
        !best || item.score > best.score ? item : best,
      null
    );

    // Temporary threshold. We will calibrate it using real images.
    const minScore = 0.18;
    const levelMargin = 0.02;

    if (
      (!bestAllowed || bestAllowed.score < minScore) &&
      (!bestHigher || bestHigher.score < minScore)
    ) {
      return {
        label: "",
        status: "unknown",
        top: candidates,
        levelCandidates
      };
    }

    if (
      bestHigher &&
      bestHigher.score >= minScore &&
      (!bestAllowed ||
        bestHigher.score > bestAllowed.score + levelMargin)
    ) {
      return {
        label: "",
        status: "higher_level",
        detectedLevel: bestHigher.level,
        top: candidates,
        levelCandidates
      };
    }

    if (bestAllowed && bestAllowed.score >= minScore) {
      return {
        label: labels[bestAllowed.index],
        status: "recognized",
        detectedLevel: bestAllowed.level,
        top: candidates,
        levelCandidates
      };
    }

    return {
      label: "",
      status: "unknown",
      top: candidates,
      levelCandidates
    };

  } finally {
    input.dispose();
    output?.dispose();
  }
}


self.onmessage = async ({ data }) => {
  try {
    if (data.type === "prepare") {
      detector = await prepareDetector(data);

      self.postMessage({
        ready: true,
        model: "CLIP ViT-B/32",
        count: detector.labels.length,
        dimensions: detector.dimensions
      });

      return;
    }

    if (data.type === "test") {
      if (!detector) {
        throw new Error("CLIP model is not initialized");
      }

      self.postMessage({
        ready: true,
        message: "CLIP model is loaded and inference test passed"
      });

      return;
    }

    if (data.type === "recognize") {
      const result = await recognizeClip(
        data.pixels,
        data.width ?? 256,
        data.height ?? 256,
        data.maxLevel ?? 3
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

