from datasets import load_dataset

from .config import DATASET_NAME, DATASET_SUBSET


def load_sentiment_dataset():
    dataset = load_dataset(
        DATASET_NAME,
        revision="refs/convert/parquet",
        data_dir=DATASET_SUBSET,
    )

    return dataset


from distilbert_sentiment.data import load_sentiment_dataset
from distilbert_sentiment.config import LABEL_NAMES

dataset = load_sentiment_dataset()

# Convert splits to pandas DataFrames for easier exploration
train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

print("Dataset Sizes:")
print(f"  Training set:   {len(train_df)} samples")
print(f"  Validation set: {len(val_df)} samples")
print(f"  Test set:       {len(test_df)} samples")

print("\nClass Distribution (Training Set):")

class_dist = train_df["label"].value_counts().sort_index()

for i, count in enumerate(class_dist):
    print(
        f"  {LABEL_NAMES[i]:10s}: "
        f"{count:4d} samples "
        f"({100 * count / len(train_df):.1f}%)"
    )