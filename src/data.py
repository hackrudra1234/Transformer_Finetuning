from datasets import load_dataset

from src.config import DATASET_NAME, DATASET_SUBSET
from src.config import LABEL_NAMES
from src.config import MAX_LENGTH


def load_sentiment_dataset():
    dataset = load_dataset(
        DATASET_NAME,
        revision="refs/convert/parquet",
        data_dir=DATASET_SUBSET,
    )

    return dataset




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


from src.config import MAX_LENGTH


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        desc="Tokenizing"
    )

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    tokenized_dataset = tokenized_dataset.rename_column(
        "label",
        "labels"
    )

    tokenized_dataset.set_format("torch")

    return tokenized_dataset