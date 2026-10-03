from src.config import LABEL_NAMES
from src.config import (
    MODEL_NAME,
    DATASET_NAME,
    DATASET_SUBSET,
    NUM_LABELS,
    LABEL_NAMES,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    MAX_LENGTH,
)
from src.data import load_sentiment_dataset, tokenize_dataset
from src.model import load_tokenizer, load_model
from src.config import MODEL_SAVE_PATH
from src.train import (
    build_trainer,
    train_model,
    evaluate_model,
    save_model,
)
from src.metrics import get_detailed_metrics
from src.inference import predict_sentiment
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME} ({DATASET_SUBSET})")
    print(f"Number of classes: {NUM_LABELS}")
    print(f"Labels: {LABEL_NAMES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print("=" * 60)

    # 1. Load dataset
    dataset = load_sentiment_dataset()
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print("\nDataset Sizes:")
    print(f"Training set:   {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    print(f"Test set:       {len(test_df)}")

    class_dist = train_df["label"].value_counts().sort_index()

    print("\nClass Distribution:")

    for i, count in enumerate(class_dist):
        print(
        f"{LABEL_NAMES[i]:10s}: "
        f"{count} "
        f"({100 * count / len(train_df):.1f}%)"
    )
    plt.figure(figsize=(6, 4))

    plt.bar(
    LABEL_NAMES,
    class_dist.values
    )

    plt.title("Class Distribution - Training Set")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Samples")

    plt.tight_layout()
    plt.show()

    # 2. Load tokenizer
    tokenizer = load_tokenizer()

    # 3. Tokenize dataset
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer
    )

    # 4. Load DistilBERT model
    model = load_model()

    # 5. Create Trainer
    trainer = build_trainer(
        model,
        tokenized_dataset
    )

    # 6. Fine-tune

    train_result = train_model(trainer)

    # 7. Evaluate on test data
    test_results = evaluate_model(
        trainer,
        tokenized_dataset
    )

    print("\nTEST RESULTS")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    # 8. Detailed evaluation
    details = get_detailed_metrics(
        trainer,
        tokenized_dataset,
        LABEL_NAMES
    )

    print("\nCLASSIFICATION REPORT")
    print(details["classification_report"])

    print("\nPER-CLASS ACCURACY")
    for label, accuracy in zip(
        LABEL_NAMES,
        details["class_accuracies"]
    ):
        print(f"{label}: {accuracy:.2%}")

    # 9. Save fine-tuned model
    save_model(
        trainer,
        tokenizer,
        MODEL_SAVE_PATH,
    )

    # 10. Test one prediction
    text = "I absolutely love this product!"

    result = predict_sentiment(
        text,
        trainer.model,
        tokenizer,
        trainer.model.device
    )

    print("\nPREDICTION")
    print(result)


if __name__ == "__main__":
    main()
