#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(module)s: %(message)s", level="INFO"
)
logger = logging.getLogger(__name__)


import json
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd

import utils


def get_pmid2cuis(path: str) -> Dict[str, List[str]]:

    # i    PMID:       PubMed abstract identifier
    # ii.  Type:       i.e., gene, disease, chemical, species, and mutation
    # iv.  Concept ID: Corresponding database identifier (e.g., NCBIGene ID, MESH ID)
    # v.   Mentions:   Bio-concept mentions corresponding to the PubMed abstract
    # vi.  Resource: Various manually annotated resources are included in the files (e.g., MeSH and gene2pubmed)
    columns = ["pmid", "type", "cui", "mentions", "resource"]

    reader = pd.read_csv(
        path,
        chunksize=500000,
        sep="\t",
        names=columns,
        na_filter=True,
        # quoting = 3,
        quotechar="~",
    )

    pmid2cuis = defaultdict(set)

    logger.info(f"Start parsing file `{path}`")

    lines = 0
    kept = 0
    for idx, chunk in enumerate(reader):

        lines += len(chunk)

        chunk = chunk[~chunk["mentions"].isna()]

        chunk = chunk[~chunk["cui"].isna()]

        kept += len(chunk)

        pmids = list(chunk["pmid"])
        cuis = list(chunk["cui"])

        for pmid, cui in zip(pmids, cuis):

            pmid2cuis[pmid].add(cui)

        if (idx + 1) % 10 == 0:
            logger.info(
                f"#PROGRESS: parsed chunk : {idx+1} : {lines} lines - {kept} kept"
            )

    logger.info(f"Completed parsing file `{path}`: {len(pmid2cuis)} unique PMIDs")

    return pmid2cuis


def create_shards(pmid2cuis: Dict[str, List[str]], outdir: str):

    items = list(pmid2cuis.items())

    shard_size = int(2e5)

    logger.info(
        f"Start splitting {len(pmid2cuis)} PMIDs with CUIs into shards in `{outdir}`"
    )

    for idx, chunk in enumerate(utils.chunkize_list(items, size=shard_size)):

        outdict = {pmid: list(cuis) for pmid, cuis in chunk}

        with open(os.path.join(outdir, f"shard{idx}.json"), "w") as outfile:

            json.dump(outdict, outfile, indent=1)

    logger.info("Completed writing shards in `{outdir}`")


if __name__ == "__main__":

    conf = utils.load_conf()

    pubtator_annotations_dir = os.path.join(conf["pubtator"]["annotations"])

    pmid2cuis = defaultdict(set)

    annotation_files = ["gene2pubtatorcentral", "disease2pubtatorcentral"]

    for file_name in annotation_files:
        pubtator_annotations_file = os.path.join(pubtator_annotations_dir, file_name)
        utils.check_exist(pubtator_annotations_file)
        tmp_pmid2cuis = get_pmid2cuis(pubtator_annotations_file)
        for pmid, cuis in tmp_pmid2cuis.items():
            pmid2cuis[pmid].update(cuis)

    outdir = os.path.join(conf["pubtator"]["annotations"], "shards")
    os.makedirs(outdir, exist_ok=True)

    create_shards(pmid2cuis=pmid2cuis, outdir=outdir)
