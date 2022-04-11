#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:48:06 2021

@author: ele
"""

import logging

logger = logging.getLogger(__name__)


import os

from flair.data import Corpus
from flair.trainers import ModelTrainer

import utils
from utils.flair_utils import MultiFileColumnCorpus, get_hunflair_tagger

if __name__ == "__main__":

    conf = utils.load_conf()

    utils.set_seed(42)

    logging.getLogger("gensim").setLevel(logging.ERROR)

    entity2model = {
        "Enhancer": "base",
        "Promoter": "base",
        "Tfbs": "base",
        "Tissue": "hunflair-cellline",
        "Gene": "hunflair-gene",
        "Disease": "hunflair-disease",
        "Species": "hunflair-species",
    }

    column_format = {0: "text", 1: "ner"}

    results_dir = os.path.join(conf["results"], "ner", "kfold")

    corpus_conll_path = conf["corpus"]["conll"]
    folds_path = os.path.join(os.getcwd(), "data", "corpus", "folds.json")
    folds = utils.load_json(folds_path)

    for entity in entity2model.keys():

        logger.info(f"Start K-FOLD run for {entity} model")

        run_dir = os.path.join(corpus_conll_path, entity)

        for idx, splits in folds.items():

            train_files = [
                os.path.join(run_dir, f"{i}.conll") for i in splits.get("train")
            ]
            test_files = [
                os.path.join(run_dir, f"{i}.conll") for i in splits.get("test")
            ]

            corpus: Corpus = MultiFileColumnCorpus(
                train_files=train_files,
                test_files=test_files,
                column_format=column_format,
            )

            tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")

            model = entity2model[entity]

            tagger = get_hunflair_tagger(name=model, tag_dictionary=tag_dictionary)

            trainer: ModelTrainer = ModelTrainer(tagger, corpus)

            base_path = os.path.join(results_dir, entity, f"fold{idx}")
            fold_results = trainer.train(
                base_path=base_path,
                train_with_dev=True,
                max_epochs=50,
                learning_rate=0.1,
                mini_batch_size=32,
                checkpoint=False,
                save_final_model=False,
            )
