# ByteNet reproduction with a position invariant statistics branch and gated fusion

Semester project by Payaldurga, Mehul and Arpita, IIIT Vadodara, under Prof. Jignesh Patel.

This repository reimplements the Byte2Image / ByteNet idea from Liu et al., "ByteNet: Rethinking multimedia file fragment classification through visual perspectives" (IEEE Trans. Multimedia, 2025), and tests one extension: replace the position dependent byte branch with a byte histogram and entropy branch, and merge the branches with a learned sigmoid gate.

## Layout

| Path | Contents |
|---|---|
| `bytenet/byte2image.py` | bit shift stack, intrabyte n-gram, `byte_stats` (histogram and entropy) |
| `bytenet/models.py` | ByteNet with ResNet or PoolFormer image branch, byte or histogram side branch, concat or gated fusion |
| `bytenet/dataset.py` | dataset, augmentation, mixup |
| `bytenet/data_prep.py` | builds `data/fragments.npz` from real files on the machine |
| `bytenet/train.py` | training, corruption robustness test, confusion matrix |
| `run_all.sh`, `run_job.sh` | runs all 24 experiments (8 configurations x 3 seeds) |
| `aggregate.py`, `mkfigs.py` | tables and figures from `results/` |
| `results/` | per run `result.json` files and `aggregate.json` from the reported runs |
| `data/fragments.npz` | the exact fragment set used in the paper (2,240 fragments, 7 classes) |
| `paper/` | manuscript source and PDF |

## Run

```bash
pip install -r requirements.txt
./run_all.sh          # about 4 minutes per run on one CPU core, 24 runs
python3 mkfigs.py
```

A single run:

```bash
python3 -m bytenet.train --name histgate_s1 --seed 1 --side hist --fusion gate --out results/histgate_s1
```

## Results (3 seeds, 448 test fragments, one standard error about 2 points)

| Configuration | Accuracy (%) |
|---|---|
| ByteResNet baseline (byte + image) | 71.9 +/- 1.2 |
| Image branch only | 73.3 +/- 1.9 |
| Byte branch only | 22.2 +/- 0.9 |
| Histogram branch only | 74.9 +/- 2.8 |
| Image + histogram | 77.2 +/- 1.7 |
| Image + histogram + gate | 77.1 +/- 3.8 |
| ByteFormer baseline | 53.2 +/- 1.9 |
| ByteFormer + histogram + gate | 78.1 +/- 1.2 |

At 30% corrupted bytes the baseline falls to 52.3% and the extension keeps 72.4%.

## Caveats

* The dataset is built from files on one machine (ELF, gzip, HTML, JSON, PDF, PNG, Python) and is not the benchmark used in the paper, so absolute numbers are not comparable with the paper. `data/fragments.npz` is included so the reported runs can be repeated exactly. Rebuilding it with `data_prep.py` on another machine gives a different dataset and different numbers.
* The histogram branch alone is within noise of the full model, and the gate gave no clear gain on clean data.
* The split is by fragment, not by file.
