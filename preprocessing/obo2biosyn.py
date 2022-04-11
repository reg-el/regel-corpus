#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(module)s: %(message)s", level="INFO"
)
logger = logging.getLogger(__name__)


import os
from collections import defaultdict
from typing import List

import obonet

import utils


def get_obo_synonyms(synonyms: List[str]) -> List[str]:

    synonyms = [s.split('"')[1] for s in synonyms]

    return synonyms


def obo2biosyn_dict(in_path: str, out_path: str):

    graph = obonet.read_obo(in_path)

    dictionary = defaultdict(set)

    for doid, data in graph.nodes(data=True):

        doid = doid.split(":")[1]
        name = data.get("name")
        synonyms = data.get("synonym")

        names = []

        if name is not None:
            names.append(name)
        if synonyms is not None:
            synonyms = get_obo_synonyms(synonyms)
            names += synonyms

        for name in names:
            dictionary[name].add(doid)

    with open(out_path, "w") as outfile:
        for name, doids in dictionary.items():
            doids = [str(i) for i in sorted(doids)]
            cui = "|".join(doids)
            outfile.write(f"{cui}||{name}\n")


if __name__ == "__main__":

    conf = utils.load_conf()

    for entity in ["disease", "tissue"]:
        inpath = conf["ontology"][entity]
        outpath = os.path.join(os.path.dirname(inpath), "train_dictionary.txt")
        logger.info(f"Start generating dictionary for {entity} BioSyn model training")
        obo2biosyn_dict(in_path=inpath, out_path=outpath)
        logger.info(f"Created BioSyn dictionary from `{inpath}` in `{outpath}`")
