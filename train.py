"""Train a small character n-gram classifier from labelled lines.

  python train.py --train train.jsonl --test test.jsonl --out models/mymodel

Rows are {"text": "...", "label": "..."}, one per line; test rows may also carry "group", which is
used only to break the result down. Writes classifier.json (labels, settings, test results) and
classifier.npz (the table) into --out. Read the model back with zo_classify.py.

The model is fastText without the hidden layer: every run of 1 to 5 characters in a line (spaces
and line ends included) is hashed to a row of one table, and the rows found in a line are averaged
into a score per label. It trains on a CPU in about a minute for 200,000 lines.

Needs numpy and torch (CPU is enough).
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from zo_classify import DEFAULTS, Classifier, featurize, report


def read(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if isinstance(r.get("text"), str) and r.get("label") not in (None, ""):
                    r["label"] = str(r["label"])
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[], help="lines kept out of training; without it one line in ten is set aside at random")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=5)
    ap.add_argument("--buckets", type=int, default=262144, help="rows of the table: the size of the model")
    ap.add_argument("--keep-case", action="store_true", help="do not lower-case the text")
    ap.add_argument("--keep-digits", action="store_true", help="do not treat every digit as the same")
    ap.add_argument("--shape", action="store_true", help="also use the shape of a line (length, punctuation, digits)")
    ap.add_argument("--no-balance", action="store_true", help="do not count a rare label for more")
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    rows, test = read(a.train), read(a.test)
    rng = np.random.default_rng(a.seed)
    if test:
        seen = {" ".join(r["text"].lower().split()) for r in test}
        rows = [r for r in rows if " ".join(r["text"].lower().split()) not in seen]
    else:
        idx = rng.permutation(len(rows))
        n = len(rows) // 10
        test, rows = [rows[i] for i in idx[:n]], [rows[i] for i in idx[n:]]
        print(f"No test file: {n} lines set aside at random. Lines from the same book or website are then on "
              "both sides, so the result looks better than it will be on new text.")
    counts = Counter(r["label"] for r in rows)
    labels = [k for k, _ in counts.most_common()]
    if len(labels) < 2:
        raise SystemExit(f"Need at least two labels in the training rows; found {len(labels)} "
                         "(training lines that are also in the test file are left out).")
    index = {lab: i for i, lab in enumerate(labels)}
    test = [r for r in test if r["label"] in index]
    opt = {**DEFAULTS, "ngram_min": a.ngram_min, "ngram_max": max(a.ngram_min, a.ngram_max), "buckets": a.buckets,
           "lowercase": not a.keep_case, "digits": not a.keep_digits, "shape": a.shape}
    print(f"{len(rows):,} training lines, {len(test):,} test lines, {len(labels)} labels")

    torch.manual_seed(a.seed)
    bag = torch.nn.EmbeddingBag(opt["buckets"], len(labels), mode="sum", sparse=True)
    torch.nn.init.zeros_(bag.weight)
    bias = torch.nn.Parameter(torch.zeros(len(labels)))
    opt_table = torch.optim.Adagrad([bag.weight], lr=a.lr)
    opt_bias = torch.optim.Adagrad([bias], lr=min(0.1, a.lr))
    weight = None if a.no_balance else torch.tensor(
        [len(rows) / (len(labels) * counts[lab]) for lab in labels], dtype=torch.float32).clamp(max=10.0)
    texts = [r["text"] for r in rows]
    target = np.array([index[r["label"]] for r in rows], dtype=np.int64)

    def forward(batch):
        r, w, o = featurize(batch, opt)
        return bag(torch.from_numpy(r), torch.from_numpy(o[:-1]), per_sample_weights=torch.from_numpy(w)) + bias

    per_epoch = math.ceil(len(rows) / a.batch)
    total, step, t0 = max(1, math.ceil(per_epoch * a.epochs)), 0, time.time()
    while step < total:
        order = rng.permutation(len(rows))
        for s in range(0, len(order), a.batch):
            if step >= total:
                break
            pick = order[s:s + a.batch]
            loss = torch.nn.functional.cross_entropy(forward([texts[i] for i in pick]), torch.from_numpy(target[pick]),
                                                     weight=weight)
            opt_table.zero_grad()
            opt_bias.zero_grad()
            loss.backward()
            opt_table.step()
            opt_bias.step()
            step += 1
            if step % 200 == 0 or step == total:
                print(f"step {step}/{total}  loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "classifier.npz", table=bag.weight.detach().numpy().astype(np.float16),
                        bias=bias.detach().numpy().astype(np.float32))
    info = {"kind": "zomi-classifier", "format": 1, "labels": labels, "options": opt,
            "trained": time.strftime("%Y-%m-%d %H:%M"), "training_lines": dict(counts), "test": None}
    (out / "classifier.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if test:                       # scored with the saved model, so the numbers are what you will get
        model = Classifier(out)
        guess = [g for g, _ in model.predict([r["text"] for r in test])]
        text, conf = report(labels, [r["label"] for r in test], guess, [r.get("group") or "-" for r in test])
        print(text)
        info["test"] = {"rows": len(test), "accuracy": round(sum(g == r["label"] for g, r in zip(guess, test)) / len(test), 4),
                        "confusion": {t: dict(c.most_common()) for t, c in conf.items() if c}}
        (out / "classifier.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Sparse invariant checks")
    main()
