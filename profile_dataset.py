"""Profile the committed dataset and write results/dataset_profile.json.

Standard library only, so it runs without the notebook stack:

    python3 profile_dataset.py

The numbers it writes are the only dataset statistics quoted in the README.
"""

import collections
import csv
import json
import os

EMOTIONS = ["Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]
DATASET = "Bangla Emotion Dataset.csv"
OUTPUT = os.path.join("results", "dataset_profile.json")


def load_rows(path=DATASET):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def profile(rows):
    n = len(rows)
    emotion_matrix = [[int(row[name]) for name in EMOTIONS] for row in rows]

    marginals = {
        name: sum(row[name] == "1" for row in rows) / n for name in EMOTIONS
    }
    majority = {name: int(marginals[name] > 0.5) for name in EMOTIONS}
    per_label_correct = sum(
        sum(1 for i, name in enumerate(EMOTIONS) if vec[i] == majority[name])
        for vec in emotion_matrix
    )
    subset_all_zero = sum(1 for vec in emotion_matrix if not any(vec)) / n

    signatures = collections.Counter(tuple(vec) for vec in emotion_matrix)
    most_common_signature, most_common_count = signatures.most_common(1)[0]

    positives = collections.Counter(sum(vec) for vec in emotion_matrix)
    texts = collections.Counter(row["Data"].strip() for row in rows)
    topics = collections.Counter(row["Topic"] for row in rows)
    domains = collections.Counter(row["Domain"] for row in rows)
    ids = [int(row["ID"]) for row in rows]

    return {
        "source_file": DATASET,
        "rows": n,
        "columns": list(rows[0].keys()),
        "unique_ids": len(set(ids)),
        "min_id": min(ids),
        "max_id": max(ids),
        "duplicate_text_rows": n - len(texts),
        "domains": dict(domains.most_common()),
        "emotion_marginals": {name: round(marginals[name], 6) for name in EMOTIONS},
        "positive_emotions_per_row": {
            str(count): rows_with for count, rows_with in sorted(positives.items())
        },
        "distinct_emotion_signatures": len(signatures),
        "most_frequent_signature": {
            "labels": [name for i, name in enumerate(EMOTIONS) if most_common_signature[i]],
            "rows": most_common_count,
        },
        "topics": {
            "n_classes": len(topics),
            "multi_label_rows": sum(1 for row in rows if "," in row["Topic"]),
            "empty_rows": sum(1 for row in rows if not row["Topic"].strip()),
            "counts": dict(topics.most_common()),
        },
        "baselines": {
            "definition": (
                "per-label majority classifier: predict each label's majority "
                "value independently (all-zero here, since no emotion exceeds 50%)"
            ),
            "per_label_accuracy": round(per_label_correct / (n * len(EMOTIONS)), 6),
            "hamming_loss": round(
                1.0 - per_label_correct / (n * len(EMOTIONS)), 6
            ),
            "subset_accuracy": round(subset_all_zero, 6),
            "most_frequent_signature_subset_accuracy": round(
                most_common_count / n, 6
            ),
        },
    }


def main():
    profile_data = profile(load_rows())
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(profile_data, handle, indent=2, ensure_ascii=False)
    print(json.dumps(profile_data["baselines"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
