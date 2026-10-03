from transformers import TrainingArguments, Trainer

from src.config import (
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    SEED,
)
import torch

from src.metrics import compute_metrics

def get_precision_config():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
    }

def get_optimizer_name():
    if torch.cuda.is_available():
        return "adamw_torch_fused"

    return "adamw_torch"

def get_training_args():
    precision = get_precision_config()
    optimizer_name = get_optimizer_name()

    training_args = TrainingArguments(
        output_dir="outputs",

        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        bf16=precision["bf16"],
        fp16=precision["fp16"],

        optim=optimizer_name,

        seed=SEED,
        report_to="none",
    )

    return training_args

def build_trainer(model, tokenized_dataset):
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    return trainer

def train_model(trainer):
    train_result = trainer.train()

    return train_result




def evaluate_model(trainer, tokenized_dataset):
    test_results = trainer.evaluate(
        tokenized_dataset["test"]
    )

    return test_results

def save_model(trainer, tokenizer, save_directory):
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)