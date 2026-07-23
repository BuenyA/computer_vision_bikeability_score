#!/usr/bin/env python3
"""Cohen's kappa between two annotators on the 50-frame pilot (option 1).

Measures the human agreement ceiling: the fine-tuned/zero-shot models cannot be
expected to exceed the agreement between two people labelling the same frames
under the same rules (dev_documentation §2).

Annotator 1 = the existing labels in dataset/test_images/labels.csv.
Annotator 2 = dataset/eval/kappa_pilot/annotator2_labels.csv, produced by a
SECOND person filling the blank annotator2_template.csv (0/1 per class).

Per-class Cohen's kappa + observed agreement, printed and appended to the dev
documentation is left to the user. Run:
    python scripts/compute_kappa.py
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ENV = ["vegetation", "water", "city"]
A1 = REPO / "dataset" / "test_images" / "labels.csv"
PILOT = REPO / "dataset" / "eval" / "kappa_pilot"
A2 = PILOT / "annotator2_labels.csv"          # <- second annotator fills this


def read(path, wanted=None):
    out = {}
    for r in csv.DictReader(open(path)):
        fn = r["filename"]
        if wanted is not None and fn not in wanted:
            continue
        try:
            out[fn] = {c: int(float(r[c])) for c in ENV}
        except (ValueError, KeyError):
            pass                                # skip unfilled rows
    return out


def cohen_kappa(a, b):
    """Binary Cohen's kappa for two 0/1 label lists."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    if not A2.exists():
        print(f"Second-annotator file not found:\n  {A2}\n"
              f"Have a 2nd person fill {PILOT/'annotator2_template.csv'} "
              f"(0/1 per class), save it as annotator2_labels.csv, then rerun.")
        return
    a2 = read(A2)
    a1 = read(A1, wanted=set(a2))
    common = sorted(set(a1) & set(a2))
    if not common:
        print("No overlapping labelled frames between the two annotators.")
        return
    print(f"Cohen's kappa on {len(common)} co-labelled pilot frames:\n")
    kappas = []
    for c in ENV:
        a = [a1[f][c] for f in common]
        b = [a2[f][c] for f in common]
        k, po = cohen_kappa(a, b)
        kappas.append(k)
        print(f"  {c:11} kappa={k:.3f}  observed_agreement={po:.3f}  "
              f"(pos: A1={sum(a)}, A2={sum(b)})")
    print(f"\n  mean kappa over classes: {sum(kappas)/len(kappas):.3f}")
    print("\nInterpretation: <0.40 poor, 0.40-0.60 moderate, 0.60-0.80 substantial, "
          ">0.80 almost perfect.\nModel F1 near mean kappa == near the human ceiling.")


if __name__ == "__main__":
    main()
