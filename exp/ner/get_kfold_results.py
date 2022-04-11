#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

import utils


def collect_results(results_dir):

    entities = [
        "Enhancer",
        "Promoter",
        "Tfbs",
        "CellLine",
        "Gene",
        "Disease",
        "Species",
    ]

    results = {}

    for entity in entities:

        results[entity] = {}
        results[entity]["precision"] = []
        results[entity]["recall"] = []
        results[entity]["f1"] = []

        for i in range(5):

            fold_result_file = os.path.join(
                results_dir, entity, f"fold{i}", "training.log"
            )

            with open(fold_result_file) as infile:

                lines = infile.readlines()

                result_line = lines[-6]

                result_line = [
                    i for i in result_line.split(" ") if i not in ["", "micro", "avg"]
                ]

                precision = float(result_line[0])
                recall = float(result_line[1])
                f1score = float(result_line[2])

                results[entity]["precision"].append(precision)
                results[entity]["recall"].append(recall)
                results[entity]["f1"].append(f1score)

    df = avg_results(results)

    return df


def avg_results(results):

    df = {}

    for run, run_results in results.items():

        # print(f"Run: {run}")

        p = run_results.get("precision")
        r = run_results.get("recall")
        f1 = run_results.get("f1")

        p_avg, p_std = np.round(np.mean(p), 2), np.round(np.std(p), 2)
        r_avg, r_std = np.round(np.mean(r), 2), np.round(np.std(r), 2)
        f1_avg, f1_std = np.round(np.mean(f1), 2), np.round(np.std(f1), 2)

        df[run.upper()] = {}
        df[run.upper()]["Precision"] = f"{p_avg} ($ pm {p_std}$)"
        df[run.upper()]["Recall"] = f"{r_avg} ($ pm {r_std}$)"
        df[run.upper()]["F1"] = f"{f1_avg} ($ pm {f1_std}$)"

    df = pd.DataFrame.from_dict(df, orient="index")

    return df


if __name__ == "__main__":

    conf = utils.load_conf()

    results_dir = os.path.join(conf["results"], "ner", "kfold")

    df = collect_results(results_dir)

    print(df.to_latex())

    with open(os.path.join(conf["results"], "ner", "kfold.tsv"), "w") as outfile:

        outfile.write(df.to_latex())
