import spacy
import random
import pandas as pd
import json

from datasets import load_dataset, Dataset
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained(
    "mrm8488/t5-base-finetuned-question-generation-ap")
SAVE_PATH_RACE = "/samples/10k_nouns.json/"

nlp = spacy.load('en_core_web_sm')


def get_question(answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'],
                            max_length=max_length)

    return tokenizer.decode(output[0]).replace("<pad> question: ", ""). replace("</s>", "")


def generate_candidates(doc):
    candidates = []
    # Heuristic for selection
    for token in doc:
        if token.pos_ == "NOUN":
            candidates.append(token.text)
    return list(set(candidates))


def generate_candidates_sent(doc):
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


def process_doc_race(article, doc, questions_per_doc=2):
    # Heuristic for answer generation
    candidates = generate_candidates(doc)
    random.shuffle(candidates)

    new_docs = []
    new_id = "SYNT.txt"

    new_docs = []

    try:
        for answer in candidates[:questions_per_doc]:
            q = get_question(answer, doc)
            # Heuristic for distractor generation
            options = generate_distractors(answer, candidates)
            new_docs.append({
                'id': new_id,
                'article': article,
                'answer': ["A", "B", "C", "D"][options.index(answer)],
                'question': q,
                'options': options
            })
    except Exception as exc:
        pass

    return new_docs


def generate_new_docs_race(example_ds, questions_per_doc=2, save=False):
    new_docs = []
    articles = list(set(example_ds["article"]))
    docs = list(nlp.pipe(articles))
    n_docs = len(docs)
    for i in range(0, n_docs):
        new_docs = [
            *new_docs, *process_doc_race(articles[i], docs[i], questions_per_doc=questions_per_doc)]

    if save is not None:
        try:
            with open(f"{save}.json", 'w') as f:
                f.write(json.dumps(new_docs))
        except Exception as exc:
            print(exc)

    print(f"Generated {len(new_docs)}")


if __name__ == '__main__':
    Dataset.from_pandas(load_dataset(
        'race', 'middle', split='train').to_pandas().head(10000)).sort('example_id')
    generate_new_docs_race(
        example_ds, questions_per_doc=4, save=SAVE_PATH_RACE)
