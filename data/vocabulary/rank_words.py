"""Rank the 26,932 WordNet candidate labels for a kids' vocabulary.

Keeps only words whose most common meaning is a physical thing (WordNet category:
artifact, food, animal, plant, body part or natural object).
Score = how common the word is in English (wordfreq Zipf, 0-8) and how concrete,
i.e. picturable, it is (Brysbaert et al. 2014 concreteness ratings, 1-5).
Usage: python rank_words.py wordnet_labels.json conc.txt current_kids_words.txt out.csv
"""
import csv, json, re, sys
from wordfreq import zipf_frequency
from nltk.corpus import wordnet as wn

PICTURABLE = {"noun.artifact", "noun.food", "noun.animal", "noun.plant", "noun.body", "noun.object"}

def category(word):
    """WordNet category of the most common noun sense (WordNet lists senses by frequency)."""
    senses = wn.synsets(word.replace(" ", "_"), pos=wn.NOUN)
    return senses[0].lexname() if senses else ""

labels_path, conc_path, current_path, out_path = sys.argv[1:5]
labels = json.load(open(labels_path, encoding="utf-8"))

conc = {}
with open(conc_path, encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        conc[p[0].lower()] = (float(p[2]), float(p[6]))  # mean concreteness, percent known

current = {}
level = None
for raw in open(current_path, encoding="utf-8"):
    t = raw.strip()
    if t.startswith("## level"):
        level = int(t.split()[-1]); continue
    if t and not t.startswith("#"):
        current[t.split("|")[0].strip()] = level

FLAG = re.compile(r"\b(gun|rifle|pistol|revolver|bomb|grenade|missile|sword|dagger|bullet|cannon|"
                  r"beer|wine|whisky|whiskey|vodka|liquor|cigar|cigarette|tobacco|pipe bomb|"
                  r"syringe|vaccine|champagne|cocktail|peace pipe|whip|coffin|casket|skull|corpse|noose|handcuff|condom|bra|underwear|bikini)\b")

rows = []
for w in dict.fromkeys(labels):
    z = zipf_frequency(w, "en")
    c, known = conc.get(w.lower(), (None, None))
    rows.append({
        "word": w,
        "zipf_frequency": round(z, 2),
        "concreteness": c if c is not None else "",
        "percent_known": known if known is not None else "",
        "in_current_list": "yes" if w in current else "no",
        "current_level": current.get(w, ""),
        "flag_for_kids": "check" if FLAG.search(w) else "",
        "category": category(w).replace("noun.", ""),
    })

def score(r):
    c = r["concreteness"] if r["concreteness"] != "" else 0
    return r["zipf_frequency"] + (c - 3) * 0.8  # frequent and picturable first

ranked = [r for r in rows if r["concreteness"] != "" and r["concreteness"] >= 4.0 and r["zipf_frequency"] >= 2.5
          and "noun." + r["category"] in PICTURABLE]
ranked.sort(key=score, reverse=True)
for i, r in enumerate(ranked, 1):
    r["rank"] = i
    if r["current_level"]:
        r["suggested_level"] = r["current_level"]
    else:
        z = r["zipf_frequency"]
        r["suggested_level"] = 1 if z >= 4.3 else 2 if z >= 3.5 else 3
    r["keep (yes/no)"] = ""

cols = ["rank", "word", "category", "zipf_frequency", "concreteness", "percent_known", "in_current_list",
        "current_level", "suggested_level", "flag_for_kids", "keep (yes/no)"]
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(ranked[:2000])
print("candidates:", len(labels), "| concrete and common:", len(ranked), "| written:", min(2000, len(ranked)))
print("current-list words inside top 1000:", sum(1 for r in ranked[:1000] if r["in_current_list"] == "yes"),
      "| top 1500:", sum(1 for r in ranked[:1500] if r["in_current_list"] == "yes"))
