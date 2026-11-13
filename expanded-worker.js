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
  const { labels, dimensions, preferredCount, format } = JSON.parse(new TextDecoder().decode(labelsBytes));
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
    return { session, labels, dimensions, inverseNorms, preferredCount, vectors };
  } finally {
    URL.revokeObjectURL(scriptUrl);
    URL.revokeObjectURL(moduleUrl);
  }
}

async function recognize(pixels) {
  const { session, labels, dimensions, inverseNorms, preferredCount, vectors } = detector;
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
    let bestIndex = -1;
    let bestScore = -Infinity;
    let familiarIndex = -1;
    let familiarScore = -Infinity;
    for (let i = 0; i < labels.length; i++) {
      const base = i * dimensions;
      let similarity = 0;
      for (let j = 0; j < dimensions; j++) similarity += vectors[base + j] * embedding[j];
      const score = similarity * inverseNorms[i] / imageNorm;
      if (score > bestScore) {
        bestIndex = i;
        bestScore = score;
      }
      if (i < preferredCount && score > familiarScore) {
        familiarIndex = i;
        familiarScore = score;
      }
    }
    // Cosine similarity is a heuristic, not a probability or a safety rating.
    if (bestScore < 0.21) return "";
    if (familiarScore >= 0.19 && bestScore - familiarScore <= 0.075) return labels[familiarIndex];
    return labels[bestIndex];
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
      self.postMessage({ label: await recognize(data.pixels) });
    }
  } catch (error) {
    self.postMessage({ error: error?.message || String(error) });
  }
};
