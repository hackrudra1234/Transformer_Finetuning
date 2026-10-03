from src.config import LABEL_NAMES
from src.data import load_sentiment_dataset, tokenize_dataset
from src.model import load_tokenizer, load_model
from src.train import (
    build_trainer,
    train_model,
    evaluate_model,
    save_model,
)
from src.metrics import get_detailed_metrics
from src.inference import predict_sentiment


def main():

    # 1. Load dataset
    dataset = load_sentiment_dataset()

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
        "./distill-sentiment-model"
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
