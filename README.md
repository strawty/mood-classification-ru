# Mood classification of short Russian texts

Classifying short free-form texts into four mood categories: `спокойное`, `меланхоличное`, `энергичное`, `тревожное`.

The point of the project was not to reach a high score, but to compare three text representations honestly on the same data and the same evaluation protocol.

## Data

- 39 tracks from a personal music library
- For each track: a free-form description of what the music evokes, written before any labels existed
- Descriptions were split into individual lines: **100 samples**
- Labels assigned in a second pass, on shuffled lines, so the label came from the text and not from the memory of the song
- Class distribution: энергичное 35, тревожное 28, меланхоличное 20, спокойное 17

Each sample keeps a `track_id`, since lines from the same track were written in one sitting and share vocabulary and mood.

## Protocol

- `GroupKFold(n_splits=5)` grouped by `track_id` — lines from one track never split across train and test
- Same classifier in all three experiments: `LogisticRegression(max_iter=1000, class_weight="balanced")`
- Only the text representation changes between experiments
- Primary metric: macro-F1 (classes are imbalanced, and accuracy would reward ignoring the small ones)

## Experiments

| # | Representation | Accuracy | Macro-F1 |
|---|---|---|---|
| 01 | TF-IDF, `char_wb`, 3–5 grams | 0.37 | 0.34 |
| 02 | `rubert-tiny2`, `[CLS]` vector | 0.38 | 0.35 |
| 03 | `rubert-tiny2`, mean pooling with attention mask | 0.40 | **0.37** |

### Per class F1

| Class | TF-IDF | `[CLS]` | Mean pooling |
|---|---|---|---|
| энергичное | 0.49 | 0.47 | 0.48 |
| тревожное | 0.28 | 0.43 | 0.48 |
| меланхоличное | 0.37 | 0.29 | 0.28 |
| спокойное | 0.20 | 0.21 | 0.25 |

## Findings

**Differences in macro-F1 are within noise.** With 100 samples and 39 groups, a gap of 0.01–0.03 is not evidence of improvement. The direction matches what is expected (mean pooling beats `[CLS]` for a model that has not been fine-tuned), but the sample is too small to call it proven.

**The per-class pattern is more informative than the average.** `тревожное` improves monotonically as the representation gets more semantic (0.28 → 0.43 → 0.48). Under TF-IDF it was most often confused with `энергичное` — those texts describe risk and danger in words that look energetic character-wise. Embeddings separate them.

**`меланхоличное` goes the other way** (0.37 → 0.28). TF-IDF handles it best, which suggests these texts carry literal lexical markers that character n-grams catch and a distributed representation blurs.

So different classes want different features. That is an argument for combining both representations rather than choosing one.

**Track-level mood labels lose information.** 26 of the 39 tracks received more than one different label across their own description lines. A single mood tag per track — the format used by streaming services — cannot represent this.

**Streaming service tags disagree with perceived mood.** Comparing the labels against Yandex Music's own mood characteristic: their `Спокойное` was labelled `меланхоличное` in 8 of 11 cases; their `Бодрое` came out `тревожное` 4 times. Their taxonomy also mixes genres (`Рок`, `Электроника`) into a mood field.

## Limitations

- 100 samples is small; the test fold is roughly 8 tracks
- All descriptions were written by one person, so the model may be learning a personal writing style rather than mood
- The same person wrote the texts and assigned the labels, which is weaker than independent annotation
- The four-class scheme did not fit the data well: grandiosity, numbness, obsession and aggression appeared in the texts and had to be forced into the nearest class

## Not tested

Fine-tuning `rubert-tiny2` on the four classes. With 39 groups this would most likely overfit, so it is deferred until the dataset is larger.

Hypothesis: fine-tuning should help `меланхоличное` and `спокойное` most, since those are the classes where the frozen representation performs worst.

## What I would do next

I would like to build a larger dataset and rerun the experiments. The scores I got are not good enough to be useful yet, and with only 100 samples a difference of 0.03 in macro-F1 is noise rather than a result. I plan to collect at least 1,000 descriptions so that differences between the methods become measurable. Every class also needs enough examples: `спокойное` had only 17, which is too few to learn from.

The most interesting thing I found is that the two representations seem to capture different signals. Melancholic texts carry their mood in specific words (тоска, одиночество, пусто, расставание), and character n-grams catch these directly. Anxious texts mostly describe a situation rather than use marker words ("something dangerous is about to happen, but I want it"). Character by character they look energetic, which is why TF-IDF sent 11 of them to `энергичное`. Embeddings capture that the situation is threatening, and `тревожное` went from 0.28 to 0.48. At the same time embeddings blur the marker words of melancholic texts into a general "sad" region shared with calm texts, so `меланхоличное` dropped from 0.37 to 0.28.

My next step is to combine both representations, so the model sees words and meaning at once, and check whether this pattern holds on a larger dataset. With 100 samples it is a hypothesis, not a finding. I would also like the texts to be labelled by more than one person, since at the moment the same person wrote them and assigned the labels.

I enjoyed the process of creating my own data and training on it, including the parts that were slow.

## Reproduce

```
01_baseline.ipynb          TF-IDF + logistic regression
02_embeddings_cls.ipynb    rubert-tiny2, [CLS]
03_embeddings_mean.ipynb   rubert-tiny2, mean pooling
```

Run in Colab, no GPU required.
