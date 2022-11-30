# Synthetic QA Generation for Multiple Choice

This repo contains the code developed for my Master's Thesis for the Language Technologies Master at UNED,
designed to be used in a Colab environment with a GPU and the RACE and EntranceExams English datasets.

It has all been condensed into two notebooks for ease of use, containing all the necessary code
to generate new synthetic data and the code to carry the experiments respectively.

You can download the necessary files from the [RACE](https://www.cs.cmu.edu/~glai1/data/race/) and [EE](http://nlp.uned.es/entrance-exams/)
sites respectively.

The first notebook `QA_MC_GEN` contains all the code necessary to generate Question-Answer-Distractor pairs from a given set of data, expected in a given format to be obtained from said dataset. The necessary preprocessing pipelines are also provided. It makes use of the [T5 model](https://huggingface.co/docs/transformers/model_doc/t5) to generate Question-Answer pairs, from which distractors are generated with several methods,
own strategy could be easily implemented by modifying them.

The second notebook `QA_MC_EVAL` contains all the code necessary to evaluate said generated pairs and demonstrate their usefulness in improving model performance and evaluating whether the synthetic data is useful enough on its own and demonstrates predictability potential.
It does so through a [BERT](https://huggingface.co/docs/transformers/model_doc/bert) pretrained model, where we finetune the model 
with different combinations of real and synthetic data. Expanded tests are carried exploring the limits of this approach in the last sections.

Most code has been developed making use of 🤗 Pipelines and hosted models.

# Experiment Scripts

Under `src/generate_data.py` we've added a script to generate the synthetic data.
You need to provide it with a destination folder.

To run it (in /src/)

```
$ python -m pip install -r requirements.txt
$ python generate_data.py save_folder n_samples questions_per_doc strategy
$ python run_experiment.py all save_folder

```

For more info you can do `python generate_data.py --help` or `python run_experiment.py --help`

# Objectives

The main objectives of this work are to evaluate methods to automatically generate Multiple Choice collections and help evaluate their contribution to improve current systems. To this end we will work towards the following objectives:

In `QA_MC_GEN`

1. Generate Question-Answer pairs from a given text.
2. Propose different methods of distractor generation for said pairs.

In `QA_MC_EVAL`

3. Evaluate how the quality and quantity of these tuples affects current systems.
4. Evaluate how the different types of distractors impact the results.
