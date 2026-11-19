# Test results

Held-out lines, 19 November 2026. "Seen source" = other documents of a website or book that was used in
training. "Unseen source" = a website or book kept out of training altogether.

## 1. Language identifier

**92.6% of 21,338 lines right.** The twelve Zo languages alone: 90.7% of 16,208.

| Kind of test line | Right | Lines |
|---|---:|---:|
| Bible | 99.2% | 2,636 |
| Other text, seen source | 94.1% | 9,944 |
| Other text, unseen source | 88.8% | 8,758 |

| Words in the line (Zo languages) | 1-3 | 4-6 | 7-12 | 13-25 | 26+ |
|---|---:|---:|---:|---:|---:|
| Right | 56.3% | 71.7% | 85.0% | 96.0% | 97.6% |

| How sure the model says it is | Share of lines | Right |
|---|---:|---:|
| 90% or more | 63% | 99.7% |
| 50% or more | 89% | 97.6% |

| Language | Code | Precision | Recall | Test lines | Mostly mistaken for |
|---|---|---:|---:|---:|---|
| Paite | pck | 80.3% | 86.3% | 1,486 | Tedim, Vaiphei, Zou |
| Tedim | ctd | 91.6% | 91.1% | 1,476 | Paite, Siyin |
| Zou | zom | 87.5% | 89.3% | 1,481 | Paite, Vaiphei |
| Simte | smt | 96.7% | 77.2% | 1,356 | Paite, Zou |
| Vaiphei | vap | 93.0% | 90.7% | 1,470 | Paite, Thadou |
| Gangte | gnb | 93.8% | 93.4% | 1,191 | Vaiphei, Zou |
| Thadou | tcz | 93.9% | 95.2% | 1,494 | Zou |
| Hmar | hmr | 93.2% | 90.7% | 1,484 | Mizo |
| Mizo | lus | 89.6% | 94.6% | 1,199 | Hmar |
| Hakha | cnh | 91.3% | 96.5% | 1,192 | Falam |
| Falam | cfm | 96.4% | 89.1% | 1,225 | Hakha |
| Siyin | csy | 95.3% | 96.1% | 1,154 | Tedim |
| English | eng | 91.4% | 99.1% | 1,500 | - |
| Hindi | hin | 100% | 100% | 1,191 | - |
| Burmese | mya | 100% | 100% | 1,160 | - |
| Meitei | mni | 100% | 100% | 600 | - |
| other | | 88.0% | 90.7% | 679 | Paite, Thadou |

### The close cousins

All test lines. Rows: the true language. Columns: what the model said.

| | Paite | Tedim | Zou | Simte | Vaiphei | another | lines |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paite | 1,282 | 38 | 22 | 13 | 28 | 103 | 1,486 |
| Tedim | 38 | 1,345 | 7 | 4 | 3 | 79 | 1,476 |
| Zou | 32 | 16 | 1,322 | 7 | 21 | 83 | 1,481 |
| Simte | 144 | 11 | 73 | 1,047 | 6 | 75 | 1,356 |
| Vaiphei | 38 | 3 | 11 | 2 | 1,334 | 82 | 1,470 |

On lines of 7 words or more: Paite 93.2%, Tedim 95.1%, Zou 95.2%, Vaiphei 95.3%, Simte 81.3% right.

Simte is the weak one. On the one Simte website held out only 47% of lines are called Simte; the model
knows Simte mostly as Bible and school books. Siyin's labels outside its Bible come from a public
tagger and are not verified.

### English, Hindi, Burmese, Meitei and "other"

- English: Bible 99.7%, Wikipedia text 100%, and the English side of Paite articles (never trained on,
  full of Zo names) 98.0%. 140 lines filed under another label were called English; of 16 read, most
  really were English lines sitting in Zo sources.
- Hindi, Burmese, Meitei: every test line right, including 244 Hindi and 151 Burmese lines of 12
  characters or fewer. Each is in its own script (Meitei in Bengali script). Not tested against other
  languages in the same scripts (Bengali, Assamese, Nepali, Marathi).
- Written in Latin letters these languages are not recognised: of 400 Meitei number phrases in Latin
  letters, none was called Meitei and 309 went to "other".
- Zo lines that quote English (160 test lines with two or more English function words): 81.2% keep
  their own language, 9.4% are called English.
- "other" was trained on eleven small Kuki-Chin languages (Khumi, Bawm, Matu, Ngawn, Zyphe, Lamkang,
  Daai, Zotung, Biate, Kom, Aimol) and scan debris. Mara, held out entirely: of 600 lines, 202 "other",
  268 Mizo, 57 Hmar.

### Sorting a mixed file

3,000 non-Bible test lines of all languages in one file: 87% of lines went into language piles (the
model at least 50% sure), and 96.4% of those were in the right pile. The rest went to `unsure.txt`.
(Measured with the earlier 13-label version of the model.)

## 2. Machine-made detector

**86.0% of 10,485 lines right.** Trained on Paite, Tedim and Mizo.

| Label | Precision | Recall | Test lines |
|---|---:|---:|---:|
| machine | 82.6% | 83.1% | 4,253 |
| natural | 88.4% | 88.0% | 6,232 |

| Kind of test line | Right | Lines |
|---|---:|---:|
| Seen source | 90.8% | 4,466 |
| Unseen source | 79.1% | 4,232 |
| Natural text of nine languages it was not trained on | 90.6% | 1,787 |

| Words in the line | 1-3 | 4-6 | 7-12 | 13+ |
|---|---:|---:|---:|---:|
| Right | 76.8% | 76.3% | 82.7% | 90.8% |

Leave one source out of training, then see how much of it is still flagged:

| Source left out | Still called machine-made |
|---|---:|
| Machine-translated short sentences (4,000 lines tested) | 88.5% |
| Machine-translated film subtitles | 88.4% |
| Machine-translated Wikipedia-style text | 39.0% |

By source, on held-out lines: film subtitles 95-97%, Wikipedia-style 97-98%, a chat model's answers
82-83%, text from a different translation model (never trained on) 88%, an unverified pair collection
72-74%. Two Tedim collections filed as machine-made are flagged only half the time (47-52%), and a set
of community-verified Paite sentences is wrongly flagged half the time (54% right); whether the model
or those labels are at fault was not checked.

## 3. Cleaner

**85.3% of 5,231 lines right on the five labels.** Every test line is from a source never trained on.

| Label | Precision | Recall | Test lines | Mostly mistaken for |
|---|---:|---:|---:|---|
| ok | 88.3% | 92.1% | 2,388 | fragment, mixed |
| mixed | 92.8% | 95.0% | 900 | ok |
| fragment | 73.1% | 75.9% | 892 | ok |
| noise | 84.1% | 71.5% | 701 | fragment, furniture |
| furniture | 76.8% | 65.4% | 350 | noise |

As a keep-or-drop decision: of the lines it keeps, 88.3% are real sentences; it keeps 92.1% of real
sentences and drops 89.8% of junk.

| Kind of line | Kept or dropped correctly | Lines |
|---|---:|---:|
| Real scan noise | 95.7% | 701 |
| Real page furniture (menus, links, running headers) | 91.7% | 350 |
| Real broken lines from scanned books | 92.2% | 667 |
| Two languages joined by rule | 95.1% | 900 |
| Sentences cut in two by rule | 39.6% | 225 |
| Real sentences | 92.1% | 2,388 |

Real sentences kept, by kind: news and blogs 86-98%, song lines 99%, chat messages 66%.

It keeps headings, titles and number lists, and it does not detect two Zo languages mixed in one line.

## 4. Text-type sorter

**84.9% of 4,919 lines right.** Seen sources 94.9%; unseen sources 74.8%.

| Type | Precision | Recall | Test lines | Mostly mistaken for |
|---|---:|---:|---:|---|
| bible | 89.0% | 92.4% | 1,000 | news |
| dictionary | 89.6% | 94.6% | 997 | chat |
| news | 77.2% | 90.5% | 989 | chat |
| song | 91.0% | 72.2% | 935 | chat, dictionary |
| chat | 79.9% | 74.2% | 998 | news |

On sources never seen: news sites in three languages 92-97%; a Bible in a language whose Bible was not
trained on 95% and 80%; word lists 92-96% and a scanned dictionary 81%; chat messages typed by people
80-93%; a chat model's long answers 16-39% (they read like articles); Hakha song lyrics 42%.

There is no "none of these" label: a blog post or a textbook still gets the nearest of the five.
