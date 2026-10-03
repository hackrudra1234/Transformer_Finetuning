import torch

from src.config import MAX_LENGTH, LABEL_NAMES, NUM_LABELS

def predict_sentiment(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[0][predicted_class].item()

    return {
        "text": text,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            LABEL_NAMES[i]: probabilities[0][i].item()
            for i in range(NUM_LABELS)
        }
    }

from transformers import pipeline


def build_pipeline(model, tokenizer, device):
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return classifier