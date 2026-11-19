"""Label the lines of a file with a small character n-gram classifier.

  python zo_classify.py lang mixed.txt
        writes mixed.lang.tsv: label, how sure (0-1), the line
  python zo_classify.py lang mixed.txt --split sorted/
        also writes one file per label (sorted/pck.txt, sorted/eng.txt ...) and sorted/unsure.txt
        for lines the model is less than --min sure of (default 0.5)
  python zo_classify.py clean book.txt --split cleaned/
        cleaned/ok.txt holds the real sentences; the other files hold what was dropped
  python zo_classify.py machine pairs.jsonl --field zo
        a .jsonl file: the text is read from the field named (default: text, then zo)
  python zo_classify.py test lang my_test.jsonl
        scores a model on rows of {"text": ..., "label": ...}

The first word is a model: a folder name inside models/ next to this file, or the path of any
folder holding classifier.json and classifier.npz (train.py writes such a folder). Files may be
.txt or .jsonl, plain or .gz. Needs Python 3.9+ and numpy only. The input is only read.

In your own code:
    from zo_classify import load
    model = load("models/lang")
    model.predict(["Na dam maw?", "Thank you very much."])   # [(label, how sure), ...]
"""

import gzip
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


MODELS = Path(__file__).resolve().parent / "models"      # put model folders here: models/lang, models/clean ...
TESTS = MODELS
RESERVED = 256
DEFAULTS = {"ngram_min": 1, "ngram_max": 5, "buckets": 262144, "lowercase": True, "digits": True,
            "shape": False, "max_chars": 400}
_DIGIT = re.compile(r"\d")
_P = 1099511628211
_M1, _M2, _M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


# ------------------------------------------------------------------ features (same as the trainer's)
def _bucket(x, edges):
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def _shape(t):
    n = max(1, len(t))
    last, first = t[-1:] or " ", t[:1] or " "
    return [
        _bucket(len(t.split()), (1, 2, 3, 4, 5, 8, 12, 20, 30)),
        _bucket(len(t), (5, 10, 20, 40, 80, 160, 320)),
        _bucket(sum(c.isdigit() for c in t) / n, (0, 0.1, 0.3)),
        _bucket(sum(c.isupper() for c in t) / n, (0, 0.05, 0.15, 0.5, 0.9)),
        _bucket(sum(not (c.isalnum() or c.isspace()) for c in t) / n, (0, 0.03, 0.08, 0.2)),
        0 if last in ".!?" else 1 if last in "\"'”’)]" else 2 if last in ",;:-" else 3 if last.isalpha() else
        4 if last.isdigit() else 5,
        0 if first.isupper() else 1 if first.islower() else 2 if first.isdigit() else 3 if first in "\"'“‘(" else 4,
        _bucket(sum(len(w) for w in t.split()) / max(1, len(t.split())), (2, 3, 4, 5, 7, 10)),
    ]


def featurize(texts, opt):
    import numpy as np

    nmin, nmax = int(opt["ngram_min"]), int(opt["ngram_max"])
    space = int(opt["buckets"]) - RESERVED
    parts, shapes = [], []
    for t in texts:
        t = " ".join(str(t).split())[:int(opt["max_chars"])]
        if opt.get("shape"):
            shapes.append(_shape(t))
        if opt.get("lowercase"):
            t = t.lower()
        if opt.get("digits"):
            t = _DIGIT.sub("0", t)
        parts.append("\x02" + t + "\x03")
    n_text = len(parts)
    lens = np.array([len(p) for p in parts], dtype=np.int64)
    codes = np.frombuffer("".join(parts).encode("utf-32-le"), dtype="<u4").astype(np.uint64)
    total = int(lens.sum())
    tid = np.repeat(np.arange(n_text, dtype=np.int64), lens)
    end = np.repeat(np.cumsum(lens), lens)
    pos = np.arange(total, dtype=np.int64)
    rows, owners = [], []
    with np.errstate(over="ignore"):
        h = np.zeros(total, dtype=np.uint64)
        for n in range(1, nmax + 1):
            if total - n + 1 <= 0:
                break
            h = h[:total - n + 1] * np.uint64(_P) + codes[n - 1:] + np.uint64(1)
            if n < nmin:
                continue
            ok = pos[:total - n + 1] + n <= end[:total - n + 1]
            g = (h[ok] ^ np.uint64((n * _M1) & 0xFFFFFFFFFFFFFFFF)) * np.uint64(_M2)
            g ^= g >> np.uint64(29)
            g *= np.uint64(_M3)
            g ^= g >> np.uint64(32)
            rows.append((g % np.uint64(space)).astype(np.int64))
            owners.append(tid[:total - n + 1][ok])
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    owners = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
    counts = np.bincount(owners, minlength=n_text)
    weights = (1.0 / np.maximum(counts, 1))[owners].astype(np.float32)
    if opt.get("shape"):
        sh = np.asarray(shapes, dtype=np.int64).reshape(n_text, -1)
        k = sh.shape[1]
        rows = np.concatenate([rows, (space + np.arange(k, dtype=np.int64) * 16 + sh).ravel()])
        owners = np.concatenate([owners, np.repeat(np.arange(n_text, dtype=np.int64), k)])
        weights = np.concatenate([weights, np.full(n_text * k, 1.0 / k, dtype=np.float32)])
        counts = counts + k
    order = np.argsort(owners, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return rows[order], weights[order], offsets


class Classifier:
    def __init__(self, path):
        import numpy as np

        self.path = Path(path)
        self.info = json.loads((self.path / "classifier.json").read_text(encoding="utf-8"))
        self.labels = self.info["labels"]
        self.opt = {**DEFAULTS, **self.info["options"]}
        z = np.load(self.path / "classifier.npz")
        self.table, self.bias = z["table"].astype(np.float32), z["bias"].astype(np.float32)

    def scores(self, texts):
        """One row of probabilities per text, in the order of self.labels."""
        import numpy as np

        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for a in range(0, len(texts), 2000):
            rows, weights, offsets = featurize(texts[a:a + 2000], self.opt)
            vals = self.table[rows] * weights[:, None]
            has = offsets[1:] > offsets[:-1]
            logits = np.zeros((len(has), len(self.labels)), dtype=np.float32)
            if vals.shape[0]:
                logits[has] = np.add.reduceat(vals, offsets[:-1][has], axis=0)
            logits += self.bias
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            out[a:a + 2000] = p / p.sum(axis=1, keepdims=True)
        return out

    def predict(self, texts):
        """[(label, how sure)] for each text."""
        p = self.scores(texts)
        best = p.argmax(axis=1)
        return [(self.labels[int(b)], float(p[i, b])) for i, b in enumerate(best)]


def load(name):
    """A model by name (lang, machine, clean, type) or by folder."""
    p = Path(name).expanduser()
    if not (p / "classifier.json").exists():
        p = MODELS / name
    if not (p / "classifier.json").exists():
        have = sorted(d.name for d in MODELS.glob("*") if (d / "classifier.json").exists()) if MODELS.exists() else []
        raise SystemExit(f"No classifier called '{name}'. Trained models here: {', '.join(have) or 'none'} "
                         f"(in {MODELS}).")
    return Classifier(p)


# ------------------------------------------------------------------ files
def read_lines(path, field=None):
    """-> the text of every non-empty line of a .txt or .jsonl file (plain or .gz)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    is_json = ".jsonl" in path.name or path.name.endswith((".json.gz", ".ndjson"))
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if is_json:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get(field) if field else (r.get("text") or r.get("zo"))
                if isinstance(t, str) and t.strip():
                    yield " ".join(t.split())
            else:
                yield line.strip()


def label_file(model, path, out=None, split=None, field=None, min_sure=0.5):
    path = Path(path)
    size = path.stat().st_size * (6 if path.suffix == ".gz" else 1)
    need = size * (2.2 if split else 1.2)
    free = shutil.disk_usage(Path(split or out or path).resolve().parent if (split or out) else path.parent).free
    if free - need < 0.5e9:
        raise SystemExit(f"Not writing: this needs about {need / 1e6:.0f} MB and only {free / 1e6:.0f} MB is free "
                         "(half a GB is always left untouched).")
    stem = path.name[:-3] if path.suffix == ".gz" else path.name
    stem = stem.rsplit(".", 1)[0]
    out = Path(out) if out else path.with_name(f"{stem}.{model.path.name}.tsv")
    files, counts, unsure = {}, Counter(), 0
    if split:
        split = Path(split)
        split.mkdir(parents=True, exist_ok=True)

    def piece(name):
        if name not in files:
            files[name] = open(split / f"{name}.txt", "w", encoding="utf-8")
        return files[name]

    with open(out, "w", encoding="utf-8") as tsv:
        batch = []

        def flush():
            nonlocal unsure
            for text, (lab, sure) in zip(batch, model.predict(batch)):
                tsv.write(f"{lab}\t{sure:.2f}\t{text}\n")
                counts[lab] += 1
                if split:
                    if sure < min_sure:
                        unsure += 1
                        piece("unsure").write(f"{lab}\t{sure:.2f}\t{text}\n")
                    else:
                        piece(lab).write(text + "\n")
            batch.clear()

        for text in read_lines(path, field):
            batch.append(text)
            if len(batch) >= 2000:
                flush()
        flush()
    for f in files.values():
        f.close()
    total = sum(counts.values())
    print(f"{total:,} lines labelled -> {out}")
    for lab, n in counts.most_common():
        print(f"  {lab:<12}{n:>10,}  {n / max(1, total) * 100:5.1f}%")
    if split:
        print(f"Split into {split}/<label>.txt; {unsure:,} lines the model was less than {min_sure:.0%} sure of "
              f"are in {split}/unsure.txt with its guess.")
    return counts


# ------------------------------------------------------------------ testing
def report(labels, truth, guess, groups=None):
    conf = {a: Counter() for a in labels}
    for t, g in zip(truth, guess):
        conf.setdefault(t, Counter())[g] += 1
    lines = [f"{len(truth):,} test lines, {sum(t == g for t, g in zip(truth, guess)) / max(1, len(truth)) * 100:.1f}% right",
             f"  {'label':<12}{'precision':>10}{'recall':>9}{'lines':>8}   most often mistaken for"]
    for lab in labels:
        tp, said, there = conf[lab][lab], sum(conf[t][lab] for t in conf), sum(conf[lab].values())
        wrong = [f"{k} {v}" for k, v in conf[lab].most_common() if k != lab][:3]
        lines.append(f"  {lab:<12}{(tp / said if said else 0) * 100:>9.1f}%{(tp / there if there else 0) * 100:>8.1f}%"
                     f"{there:>8,}   {', '.join(wrong)}")
    if groups:
        by = defaultdict(lambda: [0, 0])
        for t, g, grp in zip(truth, guess, groups):
            by[grp][0] += t == g
            by[grp][1] += 1
        lines += [f"  {k}: {a / n * 100:.1f}% of {n:,}" for k, (a, n) in sorted(by.items())]
    return "\n".join(lines), conf


def test(name, path=None):
    model = load(name)
    if not path:
        raise SystemExit("Give a test file: rows of {\"text\": ..., \"label\": ...}, one per line.")
    path = Path(path)
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    rows = [r for r in rows if r.get("label") in model.labels]
    guess = [g for g, _ in model.predict([r["text"] for r in rows])]
    text, conf = report(model.labels, [r["label"] for r in rows], guess, [r.get("group") or "-" for r in rows])
    print(text)
    return rows, guess, conf


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "test":
        test(argv[1], argv[2] if len(argv) > 2 else None)
        return
    opts = {"--split": None, "--out": None, "--field": None, "--min": "0.5"}
    rest, i = [], 0
    while i < len(argv):
        if argv[i] in opts and i + 1 < len(argv):
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if len(rest) != 2:
        raise SystemExit("Give a model and a file, for example:  python zo_classify.py lang mixed.txt --split sorted/")
    label_file(load(rest[0]), rest[1], out=opts["--out"], split=opts["--split"], field=opts["--field"],
               min_sure=float(opts["--min"]))


if __name__ == "__main__":
    main(sys.argv[1:])
