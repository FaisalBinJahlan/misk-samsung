"use strict";

let detector;

async function decompress(data, length) {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream("gzip"));
  const result = new Uint8Array(await new Response(stream).arrayBuffer());
  if (result.byteLength !== length) throw new Error("Incomplete expanded model after decompression");
  return result;
}

async function prepareDetector({ model, wasm, vocabulary, index, runtimeScript, runtimeModule, lengths }) {
  const [modelBytes, wasmBytes, labelsBytes, packedIndex] = await Promise.all(
    [model, wasm, vocabulary, index].map((data, i) => decompress(data, lengths[i])),
  );
  const { labels, dimensions, levels, format } = JSON.parse(new TextDecoder().decode(labelsBytes));
  const vectors = new Int8Array(packedIndex.buffer, packedIndex.byteOffset, packedIndex.byteLength);
  if (format !== "int8" || !labels.length || vectors.length !== labels.length * dimensions) {
    throw new Error("Invalid object index");
  }
  const inverseNorms = new Float32Array(labels.length);
  for (let i = 0; i < labels.length; i++) {
    let squaredNorm = 0;
    for (let j = i * dimensions; j < (i + 1) * dimensions; j++) squaredNorm += vectors[j] * vectors[j];
    inverseNorms[i] = 1 / Math.sqrt(squaredNorm);
  }
  const scriptUrl = URL.createObjectURL(new Blob([runtimeScript], { type: "text/javascript" }));
  const moduleUrl = URL.createObjectURL(new Blob([runtimeModule], { type: "text/javascript" }));
  try {
    importScripts(scriptUrl);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: moduleUrl };
    ort.env.wasm.wasmBinary = wasmBytes.buffer;
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    ort.env.wasm.wasmBinary = undefined;
    return { session, labels, dimensions, inverseNorms, levels: levels ?? labels.map(() => 1), vectors };
  } finally {
    URL.revokeObjectURL(scriptUrl);
    URL.revokeObjectURL(moduleUrl);
  }
}

// Below this cosine score the answer is too uncertain to show a child.
const minScore = 0.21;
// When a simpler (lower level) word scores almost as high, show it instead,
// e.g. "cup" rather than a near tie with "thermos".
const levelMargin = 0.02;

async function recognize(pixels, maxLevel = 3) {
  console.log("Maximum allowed level:", maxLevel);
  const { session, labels, dimensions, inverseNorms, levels, vectors } = detector;
  const size = 256 * 256;
  const rgb = new Float32Array(size * 3);
  for (let i = 0; i < size; i++) {
    rgb[i] = pixels[i * 4] / 255;
    rgb[i + size] = pixels[i * 4 + 1] / 255;
    rgb[i + size * 2] = pixels[i * 4 + 2] / 255;
  }
  const input = new ort.Tensor("float32", rgb, [1, 3, 256, 256]);
  const { image_embeds: output } = await session.run({ pixel_values: input });
  try {
    const embedding = output.data;
    const imageNorm = Math.hypot(...embedding);
    // let bestIndex = -1;
    // let bestScore = -Infinity;
    const levelBest = new Map();
    // The three highest-scoring words, kept for measurement (shown with ?debug=1).
    const top = [];
    for (let i = 0; i < labels.length; i++) {
      const base = i * dimensions;
      let similarity = 0;
      for (let j = 0; j < dimensions; j++) similarity += vectors[base + j] * embedding[j];
      const score = similarity * inverseNorms[i] / imageNorm;
      /*if (score > bestScore) {
        bestIndex = i;
        bestScore = score;
      }*/
      const best = levelBest.get(levels[i]);
      if (!best || score > best.score) levelBest.set(levels[i], { index: i, score });
      if (top.length < 3 || score > top[2].score) {
        top.push({ word: labels[i], level: levels[i], score });
        top.sort((a, b) => b.score - a.score);
        if (top.length > 3) top.pop();
      }
    }
    const candidates = top.map(({ word, level, score }) => ({
  word,
  level,
  score: Math.round(score * 1000) / 1000
}));

const levelCandidates = [...levelBest.entries()]
  .map(([level, candidate]) => ({
    word: labels[candidate.index],
    level,
    score: Math.round(candidate.score * 1000) / 1000
  }))
  .sort((a, b) => a.level - b.level);


    // Cosine similarity is a heuristic, not a probability or a safety rating.
    // Cosine similarity is a heuristic, not a probability or a safety rating.
const allowedCandidates = [...levelBest.entries()]
      .filter(([level]) => level <= maxLevel)
      .map(([level, candidate]) => ({
        index: candidate.index,
        level,
        score: candidate.score
      }));

    const higherCandidates = [...levelBest.entries()]
      .filter(([level]) => level > maxLevel)
      .map(([level, candidate]) => ({
        index: candidate.index,
        level,
        score: candidate.score
      }));

    const bestAllowed = allowedCandidates.reduce(
      (best, candidate) =>
        !best || candidate.score > best.score ? candidate : best,
      null
    );

    const bestHigher = higherCandidates.reduce(
      (best, candidate) =>
        !best || candidate.score > best.score ? candidate : best,
      null
    );

    // No sufficiently reliable candidate.
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

    // A higher-level object is clearly more likely.
    if (
      bestHigher &&
      bestHigher.score >= minScore &&
      (!bestAllowed || bestHigher.score > bestAllowed.score + levelMargin)
    ) {
      return {
        label: "",
        status: "higher_level",
        detectedLevel: bestHigher.level,
        top: candidates,
        levelCandidates
      };
    }

    // Show the best allowed-level word.
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
    output.dispose();
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "prepare") {
      detector = await prepareDetector(data);
      self.postMessage({ ready: true, count: detector.labels.length });
    } else {
      self.postMessage(await recognize(data.pixels, data.maxLevel ?? 3));
    }
  } catch (error) {
    self.postMessage({ error: error?.message || String(error) });
  }
};
