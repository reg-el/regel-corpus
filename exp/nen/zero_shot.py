#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(module)s: %(message)s", level="INFO"
)

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import bioc

import utils
from utils.flair_utils import Metric


def get_pmid2annotations(
    collection: bioc.BioCCollection, entity_type: str
) -> Dict[str, List[Tuple[str, str, str]]]:

    annotations = defaultdict(list)

    for d in collection.documents:
        for p in d.passages:
            for a in p.annotations:
                a_type = a.infons.get("type")
                if a_type in ["DNAMutation", "ProteinMutation", "SNP"]:
                    a_type = "VARIANT"

                a_identifier = a.infons.get("identifier")

                a_type = a_type.upper()
                if a_type == entity_type:
                    annotations[d.id].append((a.text, a_type, a_identifier))

    return annotations


def run_eval(true: bioc.BioCCollection, pred: bioc.BioCCollection) -> Metric:

    metric = Metric("NEN Evaluation")

    for entity in ["GENE", "SPECIES", "VARIANT"]:

        pmid2anns_true = get_pmid2annotations(true, entity_type=entity)
        pmid2anns_pred = get_pmid2annotations(pred, entity_type=entity)

        for did, anns_true in pmid2anns_true.items():

            anns_pred = pmid2anns_pred.get(did)

            if anns_pred is None:
                for a in anns_true:
                    metric.add_fn(a[1])
            else:

                for a in anns_pred:
                    if a in anns_true:
                        metric.add_tp(a[1])
                    else:
                        metric.add_fp(a[1])

                for a in anns_true:
                    if a not in anns_pred:
                        metric.add_fn(a[1])
                    else:
                        metric.add_tn(a[1])

    return metric


if __name__ == "__main__":

    conf = utils.load_conf()

    bioc_corpus_path = conf["corpus"]["bioc"]
    utils.check_exist(bioc_corpus_path)
    corpus_true = bioc.load(bioc_corpus_path)

    bioc_pubtator_corpus_path = conf["pubtator"]["corpus_bioc"]
    utils.check_exist(bioc_pubtator_corpus_path)
    corpus_pred = bioc.load(bioc_pubtator_corpus_path)

    metric = run_eval(true=corpus_true, pred=corpus_pred)

    print(metric)

    result_dir = os.path.join(conf["results"], "nen")
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, "zero_shot.tsv")

    with open(result_file, "w") as outfile:
        outfile.write(str(metric))
