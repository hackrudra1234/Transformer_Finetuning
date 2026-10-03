from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from src.config import MODEL_NAME, NUM_LABELS, LABEL_NAMES,MAX_LENGTH



def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.model_max_length = MAX_LENGTH

    return tokenizer




def load_model():
    id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
    label2id = {label: i for i, label in enumerate(LABEL_NAMES)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    return model



def load_saved_model(save_directory):
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    model = AutoModelForSequenceClassification.from_pretrained(
        save_directory
    )

    return model, tokenizer