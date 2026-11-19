# Zo classifiers

Four small text classifiers for Zo / Kuki-Chin languages. Each reads one line of text and gives it a label:

| Model | Question it answers | Labels |
|---|---|---|
| Language identifier | Which language is this line? | Paite, Tedim, Zou, Simte, Vaiphei, Gangte, Thadou, Hmar, Mizo, Hakha, Falam, Siyin; English, Hindi, Burmese, Meitei; other |
| Machine-made detector | Did a person write this Zo text, or a machine? | natural, machine |
| Cleaner | Is this a real sentence, or junk to drop? | ok, noise, furniture, fragment, mixed |
| Text-type sorter | What kind of text is it? | bible, news, chat, song, dictionary |

They were built to keep a text collection for low-resource languages honest: to sort mixed files into
one pile per language, to find machine-translated text that had been filed as natural, and to drop scan
debris and page furniture before a language model is trained on the text.

This repository holds the code, the method and the test results. It holds **no model files and no
training text**: see [What is not here](#what-is-not-here).

## Results in one table

All figures are from lines kept out of training (19 November 2026). Each test also holds back whole
websites and books the model never saw.

| Model | Test lines | Right | On sources never seen | Main weakness |
|---|---:|---:|---:|---|
| Language (17 labels) | 21,338 | 92.6% | 88.8% | Lines of 1-3 words: 56%. Simte outside Bible and school books |
| Machine-made | 10,485 | 86.0% | 79.1% | Calls short everyday phrases machine-made |
| Cleaner (5 labels) | 5,231 | 85.3% | 85.3% | Keeps headings and titles; drops a third of chat messages |
| Text type | 4,919 | 84.9% | 74.8% | Song lyrics and chat-model answers in languages it saw little of |

For the language identifier, length decides most of it: 96% right on lines of 13-25 words and 97.6% on
longer ones, and when the model is at least 90% sure (63% of lines) it is right 99.7% of the time.

The close cousins, on lines of 7 words or more: Paite 93.2%, Tedim 95.1%, Zou 95.2%, Vaiphei 95.3%,
Simte 81.3%.

Full tables, the confusion table for the close cousins, per-source results and the experiments are in
[RESULTS.md](RESULTS.md).

## What the tests showed that is worth knowing

- **A language identifier trained mostly on Bible text recognises Bible style, not the language.** Bible
  lines score 99%; the same languages score 87-89% on websites the model never saw. Training text was
  capped at 25% Bible per language and the test reports the three kinds of line separately.
- **A machine-text detector carries over to a similar source, not to a new kind of text.** Trained without
  one set of machine-translated short sentences it still flagged 88.5% of them. Trained without
  Wikipedia-style machine translations it flagged only 39% of those.
- **It is a file-level hint, not a sentence-level verdict.** On single short sentences it is wrong too
  often to act on: a natural greeting comes out "machine" with full confidence.
- **A catch-all "other" class is not a "none of the above" detector.** It finds 91% of the small languages
  it was shown, but a language it was never shown mostly gets the nearest relative's label (Mara: 34%
  "other", 45% Mizo).
- **Script is not language.** Hindi, Burmese and Meitei are recognised perfectly because each sits in its
  own script; written in Latin letters they are not recognised at all.

## How the models work

Each model is the fastText idea without the hidden layer:

1. Take a line, lower-case it (language model only), turn every digit into `0`, and add a start and an
   end mark.
2. Every run of 1 to 5 characters, spaces included, is hashed to one of 262,144 rows of a table. Each row
   holds one number per label.
3. Average the rows found in the line, add a bias per label, take the softmax.

That is the whole model: a table of 262,144 x (number of labels) numbers, stored in half precision. It
has no notion of meaning, only of spelling, which is what separates close languages ("uhi" / "uh hi",
"ding" / "diing", "Pathian" / "Pasian" / "Pathien"). The cleaner and the text-type sorter also get eight
coarse facts about the line (its length, how it starts and ends, the share of digits, capitals and
symbols).

Training is plain logistic regression with Adagrad, six passes over the data, about one minute on a
laptop CPU for 200,000 lines. A table four times larger and ten passes gained less than one point.

| Model | Size on disk |
|---|---:|
| Language | 7.9 MB |
| Cleaner | 2.2 MB |
| Text type | 2.1 MB |
| Machine-made | 0.7 MB |

## How the tests were built

The numbers are only worth something if the test is not more of the same text, so:

- **No test line is in training**, compared with capitals, digits and punctuation ignored.
- **Whole sources are held back.** Every test has lines from websites and books that were kept out of
  training altogether, reported separately from held-out documents of sources that were used.
- **Labels come from the source, not from an earlier model.** For language, a line is used only if its
  publisher said what language it is; lines an older checker had moved to another language are left out.
- **Machine-made and natural lines are length-matched**, so the detector cannot answer from length alone.
- **Nothing was written by a language model.** Three kinds of training rows are made by rule from real
  text and are marked as such: sentences cut in two, two languages joined in one line, and dictionary
  entries printed as "word - meaning".
- The cleaner's labels are themselves rule-made, so its score measures agreement with those rules. A
  by-eye check of 70 raw lines from sources it never saw agreed on about 60.

The settings were chosen by looking at the same tests, which flatters them slightly.

## Use

Reading a model needs Python 3.9+ and numpy only.

```
python zo_classify.py lang mixed.txt                      label every line -> mixed.lang.tsv
python zo_classify.py lang mixed.txt --split sorted/      also one file per language + unsure.txt
python zo_classify.py clean book.txt --split cleaned/     cleaned/ok.txt = the real sentences
python zo_classify.py machine pairs.jsonl --field zo      a .jsonl file, text taken from one field
python zo_classify.py test lang my_test.jsonl             score a model on your own labelled lines
```

The `.tsv` has three columns: label, how sure (0 to 1), the line. With `--split`, a line the model is
less than 50% sure of goes to `unsure.txt` with its guess instead of into a pile (change with `--min 0.7`).

```python
from zo_classify import load
model = load("models/lang")
model.predict(["A tung in Pathian in lei leh van a siam.", "Thank you very much."])
# [('pck', 0.66), ('eng', 0.9...)]   (label, how sure)
```

Model folders go in `models/` (for example `models/lang/classifier.json` and `classifier.npz`).

### Train your own

Any labelled lines will do: rows of `{"text": "...", "label": "..."}`, one per line. Needs numpy and
torch (CPU is enough).

```
python train.py --train train.jsonl --test test.jsonl --out models/mymodel
python train.py --train examples/tiny_train.jsonl --out models/tiny --epochs 30     # an 8-line toy
```

`--keep-case` and `--shape` are what the cleaner and the text-type sorter use. Give a `--test` file whose
lines come from other books and websites than the training lines; without one, one line in ten is set
aside at random and the result looks better than it will be on new text.

## Files

| File | What it is |
|---|---|
| `zo_classify.py` | Reads a model and labels files. Command-line tool and a small Python module. numpy only |
| `train.py` | Trains a model from labelled lines |
| `RESULTS.md` | Every test table |
| `examples/tiny_train.jsonl` | Eight lines, to try `train.py` |

## What is not here

- **The trained models.** They are kept at
  [huggingface.co/sensix-zo/zo-classifiers](https://huggingface.co/sensix-zo/zo-classifiers), where access
  is given on request.
- **The training text.** It is a research collection of news sites, blogs, Bible translations, school
  books and community word lists. News, blogs and scripture remain their publishers' copyright, and some
  of the school books are licensed for educational and research use only.

No licence file is included yet, so the code is "all rights reserved" until one is added.
