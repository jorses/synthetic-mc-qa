import logging
import torch
import typer

import numpy as np
import pandas as pd

from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

from datasets import ClassLabel, Dataset, load_dataset
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

# TODO: sort out version compatibilities so deprecation warning for xla_device is not present
logging.getLogger("transformers").setLevel(logging.ERROR)
CLI = typer.Typer()

logger = logging.getLogger("race_runs")
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler("race_run.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=512,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# Preprocess
MAX_LEN = 512
NUM_LABELS = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

# RACE
RACE_DIR = "/content/drive/MyDrive/TFM/RACE_DATASET/output_dir/RACE"


def append_option(question, option):
    return question.replace("_", f" {option}") if "_" in question else f"{question} {option}"


def preprocess(tokenizer, exam):
    first_sentences = [exam["article"]] * len(exam["options"])
    second_sentences = [append_option(exam["question"], option) for option in exam["options"]]

    return tokenizer(first_sentences, second_sentences, truncation=True)


def preprocess_dataset(ds, tokenizer):
    return (
        ds.map(partial(preprocess, tokenizer))
        .rename_column("answer", "labels")
        .remove_columns(
            [
                "options",
                "question",
                "article",
            ]
        )
        .cast_column("labels", ClassLabel(num_classes=4, names=["A", "B", "C", "D"]))
    )


def peak_encoding(ds, tokenizer):
    accepted_keys = ["input_ids", "attention_mask", "labels", "label" if "label" in ds[0].keys() else "labels"]

    features = [{k: v for k, v in ds[i].items() if k in accepted_keys} for i in range(10)]
    batch = DataCollatorForMultipleChoice(tokenizer)(features)
    return [tokenizer.decode(batch["input_ids"][0][i].tolist()) for i in range(4)]


def get_baseline_acc(model_name, tokenized_ds):
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, truncation=True, use_fast=True)
    training_args = TrainingArguments(
        output_dir="./baseline",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    output = trainer.predict(tokenized_ds)
    logger.info(f"Baseline accuracy for {model_name}")
    logger.info(acc(tokenized_ds["labels"], [np.argmax(x) for x in output.predictions]))


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def acc(labels, predictions):
    return sum([l == p for l, p in zip(labels, predictions)]) / len(predictions)


def run_experiment(task: str, load_path: str, model: str):
    tokenizer_bert = AutoTokenizer.from_pretrained(model, use_fast=True, do_lower_case=True, truncation=True)

    ds = pd.read_json(open(f"{load_path}/{task}.json"))

    tokenized_train = preprocess_dataset(Dataset.from_pandas(ds), tokenizer_bert)


    race_validate = load_dataset("race", "middle", split="validation")
    race_test = load_dataset("race", "middle", split="test")

    tokenized_race_test = preprocess_dataset(race_test, tokenizer_bert)
    tokenized_race_validate = Dataset.from_pandas(
        Dataset.to_pandas(preprocess_dataset(race_validate, tokenizer_bert)).head(1000)
    )

    logger.info("Proceeding to training with ")
    logger.info(f"{tokenized_race_validate.shape[0]} RACE samples")
    logger.info(f"{tokenized_train.shape[0]}")

    training_args = TrainingArguments(
        **{
            "do_train": True,
            "do_eval": True,
            "fp16": True,
            "fp16_opt_level": "O1",
            "save_total_limit": 0,
            "save_steps": 0,
            "evaluation_strategy": "steps",
            "num_train_epochs": 2,
            "per_device_eval_batch_size": 8,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5.0e-05,
            "warmup_steps": 500,
            "output_dir": "./model_runs",
            "eval_steps": 100,
        }
    )

    bert_base = Trainer(
        model=AutoModelForMultipleChoice.from_pretrained(model),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_race_validate,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer_bert),
        compute_metrics=compute_metrics,
    )

    bert_base.train()

    output = bert_base.predict(tokenized_race_test)
    accuracy = acc(tokenized_race_test["labels"], [np.argmax(x) for x in output.predictions])
    logger.info(f"Experiment yielded accuracy {accuracy}")


def prologue(task, load_path, model):
    logger.info("****************************************************")
    logger.info(f"Running experiment for {task} samples for ")
    logger.info(f"Data in {load_path} and model {model}")
    logger.info("****************************************************")


@CLI.command()
def synth(
    load_path: str,
    model: Optional[str] = typer.Argument("bert-base-uncased"),
):
    prologue("synth", load_path, model)
    run_experiment("synth", load_path, model)


@CLI.command()
def real(
    load_path: str,
    model: Optional[str] = typer.Argument("bert-base-uncased"),
):
    prologue("real", load_path, model)
    run_experiment("real", load_path, model)


@CLI.command()
def both(
    load_path: str,
    model: Optional[str] = typer.Argument("bert-base-uncased"),
):
    prologue("both", load_path, model)
    run_experiment("both", load_path, model)


@CLI.command()
def all(
    load_path: str,
    model: Optional[str] = typer.Argument("bert-base-uncased"),
):
    synth(load_path, model)
    real(load_path, model)
    both(load_path, model)


if __name__ == "__main__":
    CLI()
