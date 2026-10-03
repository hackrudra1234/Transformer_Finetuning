from transformers import AutoTokenizer

from src.config import MODEL_NAME


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return tokenizer

