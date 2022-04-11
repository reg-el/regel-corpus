#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict

import numpy as np

import utils
from utils.flair_utils import Metric


def evaluate(test_ner_file, predictions_file):

    metric = Metric("Hunflair+BioSyn EL")

    with open(test_ner_file) as infile:
        test_ner = [q.strip() for q in infile.readlines()]

    with open(predictions_file) as infile:
        predictions = json.load(infile)
        test_nen = predictions["queries"]

    # y_true = [nen_pred["mentions"][0]["golden_cui"] for nen_pred in test_nen]
    # y_pred = [nen_pred["mentions"][0]["candidates"][0]["cui"] for nen_pred in test_nen]
    # found = [int(ner_pred.split("||")[-1]) for ner_pred in test_ner]
    #
    # y_true = [y for idx, y in enumerate(y_true) if found[idx] == 1]
    # y_pred = [y for idx, y in enumerate(y_pred) if found[idx] == 1]
    #
    # p, r, f1, _ = precision_recall_fscore_support(
    #     y_true=y_true, y_pred=y_pred, average="micro"
    # )

    for ner_pred, nen_pred in zip(test_ner, test_nen):

        ner_found = int(ner_pred.split("||")[-1])

        gold_cui = nen_pred["mentions"][0]["golden_cui"]

        top1 = nen_pred["mentions"][0]["candidates"][0]["cui"]

        if not ner_found:

            metric.add_fn(gold_cui)

        else:

            if gold_cui == top1:

                metric.add_tp(gold_cui)

            else:

                metric.add_fp(top1)

    p = metric.precision()
    r = metric.recall()
    f1 = metric.f_score()

    return p, r, f1


if __name__ == "__main__":

    results = {}

    conf = utils.load_conf()

    dataset_dir = conf["corpus"]["biosyn"]
    results_dir = os.path.join(conf["results"], "nen", "kfold")

    for entity in ["CellLine", "Disease"]:

        if entity not in results:

            results[entity] = defaultdict(list)

        for fold in range(5):

            # test_queries_file = os.path.join(
            #     dataset_dir,
            #     entity,
            #     f"fold{fold}",
            #     "preprocessed_test",
            #     "test.concept",
            # )

            test_ner_file = os.path.join(
                dataset_dir,
                entity,
                f"fold{fold}",
                "preprocessed_test",
                "test.ner",
            )
            predictions_file = os.path.join(
                results_dir, entity, f"fold{fold}", "predictions_eval.json"
            )

            p, r, f1 = evaluate(
                # test_queries_file=test_queries_file,
                test_ner_file=test_ner_file,
                predictions_file=predictions_file,
            )

            results[entity]["p"].append(p)
            results[entity]["r"].append(r)
            results[entity]["f1"].append(f1)

    with open(os.path.join(conf["results"], "nen", "kfold.tsv"), "w") as outfile:

        for e in results:

            p = round(np.mean(results[e]["p"]), 2)
            r = round(np.mean(results[e]["r"]), 2)
            f1 = round(np.mean(results[e]["f1"]), 2)

            p_std = round(np.std(results[e]["p"]), 2)
            r_std = round(np.std(results[e]["r"]), 2)
            f1_std = round(np.std(results[e]["f1"]), 2)

            line = f"{e} - P: {p} ({p_std}) - R: {r} ({r_std}) - F1: {f1} ({f1_std})"

            print(line)

            outfile.write(f"{line}\n")
