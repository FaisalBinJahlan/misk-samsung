"""Find look-alike word pairs in a kids-words.txt file with the MobileCLIP-S0 text encoder.
Embeds each word exactly as tools/build_index.py does (10 photo templates, averaged).
Usage: python check_pairs.py <kids-words.txt> <model folder> <out.csv> [threshold]"""
import csv, sys
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
TEMPLATES = ["a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}.", "a photo of a small {}.",
             "a photo of a large {}.", "a bright photo of a {}.", "a dark photo of the {}.", "a blurry photo of a {}.",
             "a cropped photo of the {}.", "a photo of my {}."]
words_path, model_dir, out_path = sys.argv[1:4]
th = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
labels, prompts, levels, level = [], [], [], None
for raw in open(words_path, encoding="utf-8"):
    t = raw.strip()
    if t.startswith("## level"): level = int(t.split()[-1]); continue
    if not t or t.startswith("#"): continue
    w, _, p = (x.strip() for x in t.partition("|"))
    labels.append(w); prompts.append(p or w); levels.append(level)
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
sess = ort.InferenceSession(f"{model_dir}/text_model.onnx")
pad = tok.token_to_id("!") or 0
def embed(texts):
    out = []
    for s in range(0, len(texts), 64):
        ids = [tok.encode(x).ids[:77] for x in texts[s:s+64]]
        ids = np.array([r + [pad]*(77-len(r)) for r in ids], np.int64)
        v = sess.run(None, {"input_ids": ids})[0]
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out)
tot = sum(embed([t.format(p) for p in prompts]) for t in TEMPLATES)
tot /= np.linalg.norm(tot, axis=1, keepdims=True)
S = tot @ tot.T
iu = np.triu_indices(len(labels), 1); v = S[iu]; order = np.argsort(-v)
rows = []
for k in order:
    if v[k] < th: break
    i, j = iu[0][k], iu[1][k]
    same = "same level" if levels[i] == levels[j] else "different levels"
    rows.append([f"{v[k]:.3f}", labels[i], levels[i], prompts[i] if prompts[i] != labels[i] else "",
                 labels[j], levels[j], prompts[j] if prompts[j] != labels[j] else "", same, "", ""])
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    c = csv.writer(f)
    c.writerow(["similarity","word_a","level_a","prompt_a","word_b","level_b","prompt_b","levels","decision (keep both/remove A/remove B/add prompt)","new prompt or notes"])
    c.writerows(rows)
n = len(labels); print(f"{n} words | pairs >= {th}: {len(rows)} | >=0.95: {(v>=0.95).sum()} | >=0.92: {(v>=0.92).sum()}")
