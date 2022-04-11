#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(module)s: %(message)s", level="INFO"
)
logger = logging.getLogger(__name__)


import copy
import os
import re
from typing import Dict, List, Tuple

import bioc

import utils
from utils.flair_utils import MultiFileColumnCorpus

TOKENIZATION = re.compile("([0-9a-zA-Z]+|[^0-9a-zA-Z])")


def get_sent2anns(pred_file: str) -> Dict[Tuple[str], List[str]]:

    c = MultiFileColumnCorpus(
        train_files=[pred_file],
        column_format={0: "text", 1: "ner"},
        sample_missing_splits=False,
    )

    sent2anns = {}
    for i in range(len(c.train)):
        s = c.train[i]
        tokens = tuple([t.text for t in s.tokens])
        anns = [span.text for span in s.get_spans()]
        sent2anns[tokens] = anns

    return sent2anns


def tokens_overlap(sent_tokens: Tuple[str], text: str):

    text_tokens = TOKENIZATION.split(text)

    return all([t in text_tokens for t in sent_tokens])


def write_test_split(
    documents: List[bioc.BioCDocument], sent2anns: dict, outdir: str, entity_type: str
):

    biosyn_path = os.path.join(outdir, "test.concept")
    hunflair_path = os.path.join(outdir, "test.ner")

    with open(biosyn_path, "w") as biosyn_file, open(
        hunflair_path, "w"
    ) as hunflair_file:

        for d in documents:

            for p in d.passages:

                p = filter_annotations(passage=p, entity_type=entity_type)

                d_s2sa = {
                    s: sa for s, sa in sent2anns.items() if tokens_overlap(s, p.text)
                }

                ner_pred = [a for s, sa in d_s2sa.items() for a in sa]

                for a in p.annotations:
                    start = a.total_span.offset
                    end = a.total_span.end
                    text = a.text
                    cui = a.infons.get("identifier")

                    if cui is not None:
                        cui = cui.split(":")[0]
                        cui = cui.replace(";", "|")
                        mention = f"{d.id}||{start}||{end}||{text}||{cui}\n"
                        biosyn_file.write(mention)

                        found = int(a.text in ner_pred)
                        hunflair_pred = f"{d.id}||{start}||{end}||{text}||{found}\n"
                        hunflair_file.write(hunflair_pred)


def filter_annotations(passage: bioc.BioCPassage, entity_type: str) -> bioc.BioCPassage:

    annotations = [
        a
        for a in passage.annotations
        if a.infons.get("type").upper() == entity_type.upper()
    ]

    passage.annotations = annotations

    return passage


def write_train_split(
    documents: List[bioc.BioCDocument], outdir: str, entity_type: str
):

    biosyn_path = os.path.join(outdir, "train.concept")

    with open(biosyn_path, "w") as biosyn_file:

        for d in documents:

            for p in d.passages:

                p = filter_annotations(passage=p, entity_type=entity_type)

                for a in p.annotations:
                    start = a.total_span.offset
                    end = a.total_span.end
                    text = a.text
                    cui = a.infons.get("identifier")

                    if cui is not None:
                        cui = cui.split(":")[0]
                        cui = cui.replace(";", "|")
                        mention = (
                            f"{d.id}||{start}|{end}||{entity_type}||{text}||{cui}\n"
                        )
                        biosyn_file.write(mention)


if __name__ == "__main__":

    conf = utils.load_conf()

    bioc_corpus_path = conf["corpus"]["bioc"]
    utils.check_exist(bioc_corpus_path)

    documents = bioc.load(bioc_corpus_path).documents

    folds_file = os.path.join(os.getcwd(), "data", "corpus", "folds.json")
    folds = utils.load_json(folds_file)

    biosyn_corpus_path = conf["corpus"]["biosyn"]

    ner_results_dir = os.path.join(conf["results"], "ner", "kfold")
    utils.check_exist(ner_results_dir)

    for entity_type in ["Disease", "Tissue"]:

        for idx, fold in folds.items():

            tag_documents = copy.deepcopy(documents)

            fold_dir = os.path.join(biosyn_corpus_path, entity_type, f"fold{idx}")

            train = [d for d in tag_documents if d.id in fold.get("train")]
            train_dir = os.path.join(fold_dir, "preprocessed_train")
            os.makedirs(train_dir, exist_ok=True)
            write_train_split(
                documents=train, outdir=train_dir, entity_type=entity_type
            )

            test = [d for d in tag_documents if d.id in fold.get("test")]
            test_dir = os.path.join(fold_dir, "preprocessed_test")
            os.makedirs(test_dir, exist_ok=True)

            fold_test_ner_pred = os.path.join(
                ner_results_dir, f"{entity_type}", f"fold{idx}", "test.tsv"
            )

            sent2ann = get_sent2anns(fold_test_ner_pred)

            write_test_split(
                documents=test,
                outdir=train_dir,
                sent2ann=sent2ann,
                entity_type=entity_type,
            )
