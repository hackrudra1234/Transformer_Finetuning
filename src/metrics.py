import numpy as np
import evaluate
from sklearn.metrics import classification_report, confusion_matrix


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Args:
        eval_pred: Contains predictions and labels

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1_weighted = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted"
    )

    f1_macro = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1_weighted": f1_weighted["f1"],
        "f1_macro": f1_macro["f1"]
    }


def get_detailed_metrics(trainer, tokenized_dataset, label_names):
    predictions_output = trainer.predict(
        tokenized_dataset["test"]
    )

    predictions = np.argmax(
        predictions_output.predictions,
        axis=-1
    )

    labels = predictions_output.label_ids

    report = classification_report(
        labels,
        predictions,
        target_names=label_names
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "class_accuracies": class_accuracies,
    }