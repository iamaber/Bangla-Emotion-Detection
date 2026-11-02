# Bangla Emotion Detection

Multi-label emotion classification and single-label topic classification on
Bangla social-media comments, with three comparable models: TF-IDF + XGBoost,
a BiLSTM with attention, and a fine-tuned `csebuetnlp/banglabert_small`.

This is a research prototype. The numbers below are small-sample results from a
single 15% test split, not a production benchmark.

## Dataset

`Bangla Emotion Dataset.csv` — 20,642 rows, 10 columns.

| Column | Meaning |
|---|---|
| `ID` | Row identifier from the source corpus (20,642 unique; max 28,164, so this is a subset of a larger collection) |
| `Data` | Raw Bangla comment |
| `Love`, `Joy`, `Surprise`, `Anger`, `Sadness`, `Fear` | Binary emotion labels (multi-label) |
| `Topic` | Single topic label, 1 of 12 classes |
| `Domain` | Source platform: YouTube 15,942 / Facebook 4,310 / Twitter 390 |

Emotion prevalence (multi-label, so rows need not sum to 1):

| Emotion | Share of rows |
|---|---|
| Joy | 44.3% |
| Sadness | 25.0% |
| Love | 20.2% |
| Anger | 19.7% |
| Surprise | 4.8% |
| Fear | 1.8% |

Topics: Personal 5,481 · Politics 3,370 · Entertainment 3,064 · Sports 2,596 ·
Music 2,283 · Education 2,037 · Business 583 · Food 442 · Travel 412 · Game 225 ·
Fashion 96 · Health 53.

Most rows carry one emotion (17,435); 3,135 carry two, 70 carry three and 2 carry
four. There are 38 distinct emotion combinations and 70 duplicate comment texts.

**Provenance and licence are NOT established.** The comments were scraped from
YouTube, Facebook and Twitter, whose terms and the original authors' copyright
still apply; no upstream source, collection date or annotation guideline is
recorded in this repository. See `DATA.md`. Treat the CSV as "all rights
reserved / research use only" until provenance is documented.

## Evaluation protocol

All three notebooks share `evaluation.py`, so their numbers are comparable:

* **70 / 15 / 15 multi-label-stratified split** (one split, shared by the
  emotion and topic targets). Label combinations rarer than 5 rows are pooled
  into one stratum so the split stays valid.
* **train** fits the vectorizer/tokenizer and the weights; **validation**
  selects the epoch (early stopping) and the per-label decision thresholds;
  **test** is scored exactly once and is never used for tuning.
* Vectorizers and tokenizers are fitted on the **training split only**, inside
  a `Pipeline` for the sklearn models.
* Metrics are reported by definition, never as a bare "accuracy":
  per-label precision/recall/F1, macro-F1, micro-F1, subset accuracy (all six
  labels must match) and hamming loss for emotions; accuracy, macro-F1 and
  per-class scores for the single-label topic target.

Reproduce the dataset statistics quoted above with:

```bash
python3 profile_dataset.py   # writes results/dataset_profile.json
```

## Baseline

The trivial per-label majority classifier (no features) predicts each label's
majority value. Because no emotion exceeds 50% prevalence, that is all-zero for
all six labels. On the test split:

| Baseline | Value |
|---|---|
| Per-label accuracy | 0.807 |
| Hamming loss | 0.193 |
| Subset accuracy (all six correct) | 0.000 |
| Most frequent single label signature (`Joy` only) subset accuracy | 0.336 |

Any model must beat **0.807 per-label accuracy** and **0.193 hamming loss** to be
worth anything. As the results show, none of the three does.

Source: `results/dataset_profile.json`.

## Results

Test split, 15% held out (3,097 rows). All values come from the JSON artifacts
in `results/`.

### Emotions (multi-label)

| Model | per-label acc | macro-F1 | micro-F1 | subset acc | hamming loss |
|---|---|---|---|---|---|
| Majority baseline | **0.807** | — | — | 0.000 | **0.193** |
| XGBoost + TF-IDF | 0.782 | **0.408** | 0.539 | 0.251 | 0.218 |
| BiLSTM + attention | 0.739 | 0.382 | 0.512 | 0.193 | 0.261 |

Per-label F1 (XGBoost): Love 0.45 · Joy 0.67 · Surprise 0.16 · Anger 0.42 ·
Sadness 0.58 · Fear 0.17.

Per-label F1 (BiLSTM): Love 0.44 · Joy 0.68 · Surprise 0.14 · Anger 0.43 ·
Sadness 0.58 · Fear 0.03.

**Both emotion models are worse than the trivial baseline.** Per-label accuracy
(0.782 and 0.739) is below the baseline's 0.807, and hamming loss (0.218 and
0.261) is above its 0.193. That is the same picture the original notebooks
showed once the metric is named correctly: the old headline "accuracy" of
0.5597 was this per-label binary accuracy, not exact-match accuracy, and it was
never compared against the majority floor. The models buy recall on frequent
classes by over-predicting, which is why macro-F1 looks moderate while
per-label accuracy is poor. `Surprise` (4.8%) and `Fear` (1.8%) are effectively
unlearned.

### Topic (single-label, 12 classes)

| Model | accuracy | macro-F1 |
|---|---|---|
| XGBoost + TF-IDF | 0.439 | 0.283 |
| BiLSTM + attention | **0.562** | **0.332** |

Chance level is 1/12 ≈ 0.083, and predicting `Personal` for every row gives
5,481/20,642 ≈ 0.266. Topic is the better-supported task of the two.

### BanglaBERT

The BanglaBERT notebook was **not run** for this revision: no checkpoint
download is performed during maintenance runs. It uses the same protocol and
will write `results/banglabert_emotions.json` and `results/banglabert_topic.json`
when run. No BanglaBERT numbers are claimed here.

## Reproducing

Requires CPython 3.10 (TensorFlow 2.15.1 pins Keras 2.x).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# TF/numpy determinism: PYTHONHASHSEED only applies at interpreter start
PYTHONHASHSEED=42 jupyter lab
```

Run the notebooks in this order:

1. `XGBoost.ipynb` — fast, no GPU needed
2. `Multi-Task-Neural network.ipynb` — BiLSTM, CPU is fine (~1 min/epoch)
3. `Neural network with banglabert_small.ipynb` — needs the Hub download and
   `torch` (the checkpoint is converted from PyTorch with `from_pt=True`)

Each notebook writes its metrics to `results/*.json`, its weights/tokenizer to
`results/`, and leaves the committed notebook output-free. Run tests with:

```bash
python -m unittest discover -s tests -v
# or
make test
```

`make profile` regenerates the dataset statistics, `make notebooks` executes all
three notebooks headlessly, and `make strip` removes outputs (see
`.gitattributes`, which wires `nbstripout` as a clean filter).

### Reproducibility notes

* `PYTHONHASHSEED` must be set **before** the interpreter starts; setting it in
  code has no effect on the current process.
* `evaluation.set_global_seed()` seeds Python, NumPy and
  `tf.keras.utils.set_random_seed`.
* XGBoost uses `random_state=SEED`; re-running the committed XGBoost notebook
  reproduces its metrics exactly.
* The BanglaBERT checkpoint is pinned by commit SHA so the weights cannot drift.
* Numbers are written to `results/` as artifacts and quoted from there, not
  copied out of notebook output.

## Repository layout

```
evaluation.py                     shared split, metrics, baselines, seeding
profile_dataset.py                dataset stats -> results/dataset_profile.json
XGBoost.ipynb                     TF-IDF + XGBoost baseline
Multi-Task-Neural network.ipynb    BiLSTM + attention, two heads
Neural network with banglabert_small.ipynb   BanglaBERT, two heads
tests/test_evaluation.py          protocol and padding-mask regression tests
results/                          committed metrics JSON + gitignored weights
requirements.txt                  pinned dependency set (Python 3.10)
DATA.md                           data provenance and licensing notes
Makefile                          setup / test / profile / notebooks / strip
```

## Licence

Code: MIT (`LICENSE`). Data and the pre-trained checkpoint are licensed
separately — see `DATA.md`.
