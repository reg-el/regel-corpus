#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

"""

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(module)s: %(message)s", level="INFO"
)
logger = logging.getLogger(__name__)


import os
from typing import Dict, List

from flair.data import Corpus, Sentence
from flair.models import SequenceTagger

import utils
from utils.flair_utils import Metric, MultiFileColumnCorpus


def update_metric(
    metric: Metric, sents_true: List[Sentence], sents_pred: List[Sentence]
):

    for sent_true, sent_pred in zip(sents_true, sents_pred):

        spans_true = [
            (span.tag.upper(), span.text, span.position_string)
            for span in sent_true.get_spans()
        ]

        spans_pred = [
            (span.tag.upper(), span.text, span.position_string)
            for span in sent_pred.get_spans()
        ]

        for span in spans_true:
            if span in spans_pred:
                metric.add_tp(span[0])
            else:
                metric.add_fn(span[0])

        for span in spans_pred:
            if span not in spans_true:
                metric.add_fp(span[0])


def run_hunflair_zeroshot(
    corpus_path: str, metric: Metric, column_format: Dict = {0: "text", 1: "ner"}
):

    for entity in ["Gene", "Disease", "Species"]:

        tagger = SequenceTagger.load(f"hunflair-{entity.lower()}")

        run_dir = os.path.join(corpus_path, entity)

        corpus: Corpus = MultiFileColumnCorpus(
            train_files=list(utils.iglob(run_dir, ext="conll")),
            column_format=column_format,
            sample_missing_splits=False,
        )

        sents_true = [corpus.train[i] for i in range(len(corpus.train))]

        sents_pred = [Sentence(s.to_original_text()) for s in sents_true]

        tagger.predict(sentences=sents_pred)

        update_metric(metric=metric, sents_true=sents_true, sents_pred=sents_pred)


def run_pubtator_zeroshot(
    corpus_path: str,
    pubtator_corpus_path: str,
    metric: Metric,
    column_format: Dict = {0: "text", 1: "ner"},
):

    corpus_true_path = os.path.join(os.getcwd(), "data", "corpus", "conll", "Variant")
    train_files = list(utils.iglob(corpus_true_path, ext="conll", ordered=True))
    corpus_true: Corpus = MultiFileColumnCorpus(
        train_files=train_files,
        column_format=column_format,
        sample_missing_splits=False,
    )

    corpus_pred_path = os.path.join(
        os.getcwd(), "data", "corpus", "pubtator", "conll", "Variant"
    )
    train_files = list(utils.iglob(corpus_pred_path, ext="conll", ordered=True))
    corpus_pred = MultiFileColumnCorpus(
        train_files=train_files,
        column_format=column_format,
        sample_missing_splits=False,
    )

    idxs = [
        i
        for i in range(len(corpus_true.train))
        if len(corpus_true.train[i].get_spans()) > 0
    ]

    sents_true = [corpus_true.train[i] for i in idxs]
    sents_pred = [corpus_pred.train[i] for i in idxs]

    update_metric(metric=metric, sents_true=sents_true, sents_pred=sents_pred)


if __name__ == "__main__":

    logger.info(
        "NER Zero-shot: HunFlair on GENE, DISEASE, SPECIES and tmVar on VARIANT"
    )

    conf = utils.load_conf()

    metric = Metric("NER zero-shot")

    logger.info("Start computing performance of HunFlair zero-shot")

    run_hunflair_zeroshot(corpus_path=conf["corpus"]["conll"], metric=metric)

    logger.info("Start computing performance of tmVariant")
    run_pubtator_zeroshot(
        corpus_path=conf["corpus"]["conll"],
        pubtator_corpus_path=conf["pubtator"]["corpus_conll"],
        metric=metric,
    )

    print(metric)

    results_dir = os.path.join(conf["results"], "ner")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "zero_shot.tsv"), "w") as outfile:
        outfile.write(str(metric))
