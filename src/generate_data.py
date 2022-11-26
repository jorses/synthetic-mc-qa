import json
import random
import os
import logging

import typer

from tqdm import tqdm
from enum import Enum
from importlib import import_module
from spacy.cli import download
from typing import Optional


from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# TODO: sort out version compatibilities so deprecation warning for xla_device is not present
logging.getLogger("transformers").setLevel(logging.ERROR)


class Strategy(str, Enum):
    nouns = "NOUNS"
    words = "WORDS"
    sentences = "SENTENCES"


CLI = typer.Typer()

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
SAVE_PATH_RACE = "/samples/10k_nouns.json/"


def load_model(model_name: str, **kwargs):
    try:
        model = import_module(model_name)
    except ModuleNotFoundError:
        download(model_name)
        model = import_module(model_name)

    return model.load(**kwargs)


nlp = load_model("en_core_web_sm")


def get_question(answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors="pt")

    output = model.generate(
        input_ids=features["input_ids"], attention_mask=features["attention_mask"], max_length=max_length
    )

    return tokenizer.decode(output[0]).replace("<pad> question: ", "").replace("</s>", "")


def generate_candidates_nouns(doc):
    candidates = []
    # Heuristic for selection
    for token in doc:
        if token.pos_ == "NOUN":
            candidates.append(token.text)
    return list(set(candidates))


def generate_candidates_words(doc):
    candidates = []
    # Heuristic for selection
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "VERB":
            candidates.append(token.text)
    return list(set(candidates))


def generate_candidates_sentences(doc):
    candidates = []
    # Heuristic for selection
    for sent in doc.sents:
        candidates.append(str(sent))
    return list(set(candidates))


def generate_distractors(answer, candidates):
    candidates = [c for c in candidates if c != answer]
    options = [answer, *random.sample(candidates, 3)]
    random.shuffle(options)
    return options


def process_doc_race(article, doc, questions_per_doc, strategy):
    # Heuristic for answer generation

    candidates = globals()[f"generate_candidates_{strategy.lower()}"](doc)
    random.shuffle(candidates)
    new_docs = []
    new_id = "SYNT.txt"

    try:
        for answer in candidates[:questions_per_doc]:
            q = get_question(answer, doc)
            # Heuristic for distractor generation
            options = generate_distractors(answer, candidates)
            new_docs.append(
                {
                    "id": new_id,
                    "article": article,
                    "answer": ["A", "B", "C", "D"][options.index(answer)],
                    "question": q,
                    "options": options,
                }
            )
    except Exception as exc:
        print(exc)
        pass

    return new_docs


@CLI.command()
def race(
    save: str,
    n_race_docs: Optional[int] = typer.Argument(8),
    questions_per_doc: Optional[int] = typer.Argument(5),
    strategy: Optional[Strategy] = typer.Argument(Strategy.nouns),
    initial_id: Optional[int] = typer.Argument(0),
):
    print("***********************")
    print("Loading RACE dataset...")

    df = load_dataset("race", "middle", split="train").sort("example_id").to_pandas().reset_index()

    # if not enough, take from the front
    articles = (df["article"].unique().tolist() * 2)[initial_id : (n_race_docs + initial_id)]

    df = df.head(df[df["article"] == articles[-1]].index[-1])
    docs = list(nlp.pipe(articles))

    print(f"Generating synthetic data from {n_race_docs} RACE samples")
    print(f"At a rate of {questions_per_doc} questions-answer-distractor tuples per doc")
    print(f"Following the strategy {strategy}")

    new_docs = []
    for article, doc in tqdm(zip(articles, docs)):
        new_docs = [*new_docs, *process_doc_race(article, doc, questions_per_doc, strategy)]

    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)

        try:
            with open(f"{save}/real.json", "w") as f:
                f.write(json.dumps(json.loads(df.to_json(orient="records"))))
            with open(f"{save}/both.json", "w") as f:
                f.write(json.dumps([*json.loads(df.to_json(orient="records")), *new_docs]))
            with open(f"{save}/synth.json", "w") as f:
                f.write(json.dumps(new_docs))
        except Exception as exc:
            print(exc)

    print(f"Generated {len(new_docs)} synthetic samples")
    print(f"From {df.shape[0]} RACE rows, {n_race_docs} unique documents")


if __name__ == "__main__":
    CLI()
