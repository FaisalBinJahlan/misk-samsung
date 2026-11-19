# Vocabulary Data

Easy English uses MobileCLIP-S0 (Apple) without retraining. The app compares the camera image with a fixed list of English words and shows the closest one. The quality of this list decides what the app can recognise and how often it confuses similar objects, so preparing it was the main data task of the project.

This document explains where the words come from, how they were filtered, how age levels were set, and how look-alike words were handled.

## Final list

| | Words |
|---|---|
| Level 1 (ages 4–6) | 567 |
| Level 2 (ages 7–9) | 323 |
| Level 3 (ages 10–11) | 51 |
| **Total** | **941** |

318 words have a short description (a *prompt*) that the model reads instead of the bare word, for example `puppy | small baby puppy`. The child always sees the word before the `|`.

## Files

| File | Content |
|---|---|
| `tools/kids-words.txt` | The final list, read by `tools/build_index.py` |
| `data/vocabulary/word_table.csv` | Every word with its source, age-of-acquisition rating, level and prompt |
| `data/vocabulary/lookalike_changes.csv` | Words removed or given a prompt in the look-alike check, with the reason |
| `data/vocabulary/pairs_after.csv` | Word pairs that are still similar after the check (similarity ≥ 0.90) |
| `data/vocabulary/rank_words.py` | Ranks WordNet labels by frequency and concreteness |
| `data/vocabulary/check_pairs.py` | Finds look-alike word pairs with the MobileCLIP-S0 text encoder |

## Selection rules

Every word was checked against four rules:

1. The object is found at home (toys included) or is common in children's picture books.
2. General words come first. Common specific types are allowed (for example *school bus*) when they look clearly different.
3. Places are allowed only when they are common in children's picture books (for example *beach*, *farm*, *desert*).
4. The word is safe for children: no weapons, alcohol, drugs or sensitive body parts.

The first version of the rules was stricter: home objects only, general words only, and no places. Rules 1 to 3 were widened in steps to reach a list of about 1,000 words. The effect of each step is shown below.

## Steps

| Step | Source and method | Added | Total |
|---|---|---|---|
| 1 | WordNet 3.0 labels used by the first prototype | | 26,932 candidates |
| 2 | Kept physical things (WordNet category of the most common sense), concreteness ≥ 4 out of 5, and common words; ranked by frequency and concreteness | | 3,139 candidates |
| 3 | Four rules (strict version) applied to the 3,139 ranked labels | 409 | 409 |
| 4 | Basic home words the ranking missed, such as *book*, *table* and *glass*, and the objects used in the test scenarios | 46 | 455 |
| 5 | Home objects from the MacArthur-Bates CDI, Words & Sentences form | 38 | 493 |
| 6 | Words from the Kuperman list: learned by age 11 (or 13 with a lower concreteness limit), concreteness ≥ 4.3, any physical WordNet sense; about 3,500 words reviewed | 189 | 682 |
| 7 | Rule 1 widened to common picture-book words (sun, tree, ship, lion) | 152 | 834 |
| 8 | Rule 3 widened to picture-book places, and rule 2 widened to common specific types | 171 | 1,005 |
| 9 | Look-alike check: 64 words removed, 139 prompts added or changed | −64 | **941** |

Some words added in steps 3 to 8 were later removed in step 9, so the numbers by source in `word_table.csv` are lower than the numbers added.

## Age levels

Levels come from the age-of-acquisition ratings of Kuperman, Stadthagen-Gonzalez and Brysbaert (2012), which give the average age at which a word is learned:

- level 1: up to age 6
- level 2: ages 7 to 9
- level 3: age 10 and above

For words of two or more parts, the hardest part decides the level. Words without a rating were given a level by hand, or level 2 by default; `word_table.csv` marks these.

The ratings come from American adults. Three foods common in Saudi homes (*hummus*, *pomegranate*, *kebab*) were moved to level 2 by hand.

207 words in the final list also appear in the CDI form, which records words produced by children aged 16 to 30 months. These words are early-learned by direct evidence from children, not only from adult ratings.

## Look-alike check

The model's text encoder was used to measure how similar every pair of words is, in the same way `tools/build_index.py` builds the index (ten photo prompts per word, averaged). Pairs with a similarity of 0.90 or more were reviewed:

- Same object with two names (*sea* and *ocean*): one word removed.
- A type that looks the same as the general word (*orange juice* and *juice*): the type removed.
- Different objects that look similar (*dog* and *puppy*): a prompt added that describes the visible difference (*adult dog*, *small baby puppy*).
- Body parts (*foot* and *toe*): kept, because they are important for children, although the model may confuse them.

| | Before | After |
|---|---|---|
| Words | 1,005 | 941 |
| Pairs with similarity ≥ 0.90 | 178 | 38 |
| Pairs with similarity ≥ 0.92 | 91 | 14 |
| Pairs with similarity ≥ 0.95 | 24 | 0 |

## Limitations

- The selection rules involve human judgement, and another reviewer could decide differently for some words.
- The age levels depend on ratings from another culture.
- Similarity between words is only a guide. A lower text similarity does not guarantee that the model separates the two objects in a photo.
- Places and outdoor objects can only be shown as pictures in a book, which is harder for the model than real objects.
- An object that is not in the list may receive the closest wrong word, so the confidence threshold must be tuned again for this list.

## How the list will be tested

The same photos will be tested with the original 1,000-word list, the 493-word list (strict rules) and this 941-word list. We will compare label accuracy by level and the number of wrong labels given to objects that are not in the list.

## Rebuilding the index

After any change to `tools/kids-words.txt`, rebuild the index so the app uses the new list:

```
python tools/build_index.py <folder with text_model.onnx and tokenizer.json>
```

The two model files come from [Xenova/mobileclip_s0](https://huggingface.co/Xenova/mobileclip_s0) (`onnx/text_model.onnx` and `tokenizer.json`). Do not commit them; they are only needed to build the index.

## Sources

- WordNet 3.0. Princeton University. https://wordnet.princeton.edu
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods, 46*, 904–911.
- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. *Behavior Research Methods, 44*, 978–990. https://doi.org/10.3758/s13428-012-0210-4
- Frank, M. C., Braginsky, M., Yurovsky, D., & Marchman, V. A. (2017). Wordbank: an open repository for developmental vocabulary data. *Journal of Child Language, 44*(3), 677–694. Data: MacArthur-Bates CDI, Words & Sentences (English, American). https://wordbank.stanford.edu
- Speer, R. *wordfreq* (Python library). https://github.com/rspeer/wordfreq
- MobileCLIP-S0, Apple, used under the Apple Machine Learning Research Model License; ONNX version from Xenova/mobileclip_s0.

The rating files from Kuperman et al. and Wordbank are not stored in this repository; download them from the links above.
