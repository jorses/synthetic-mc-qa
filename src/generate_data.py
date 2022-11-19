import json
import random
import os

import typer

from tqdm import tqdm
from enum import Enum
from importlib import import_module
from spacy.cli import download

from datasets import load_dataset, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
def race(save: str, n_race_samples: int = 8, questions_per_doc: int = 5, strategy: Strategy = Strategy.nouns):
    ds = Dataset.from_pandas(load_dataset("race", "middle", split="train").to_pandas().head(n_race_samples)).sort(
        "example_id"
    )

    articles = list(set(ds["article"]))
    docs = list(nlp.pipe(articles))

    print(f"Generating synthetic data from {n_race_samples} RACE samples")
    print(f"At a rate of {questions_per_doc} questions-answer-distractor tuples per doc")
    print(f"Following the strategy {strategy}")

    new_docs = []
    for article, doc in tqdm(zip(articles, docs)):
        new_docs = [*new_docs, *process_doc_race(article, doc, questions_per_doc, strategy)]

    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)

        try:
            with open(f"{save}/all_data.json", "w") as f:
                f.write(json.dumps([*json.loads(ds.to_pandas().to_json(orient="records")), *new_docs]))
            with open(f"{save}/synthetic_data.json", "w") as f:
                f.write(json.dumps(new_docs))
        except Exception as exc:
            print(exc)

    print(f"Generated {len(new_docs)}")


if __name__ == "__main__":
    CLI()
