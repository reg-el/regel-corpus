#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

import os
import re
import glob
import bioc
import json
import requests
import pickle
import random
import numpy as np
from omegaconf import OmegaConf
import torch
from typing import List, Optional, Iterable, Any

def chunkize_list(l : List[Any], size : int = 10) -> Iterable[List[Any]]:
    
    for i in range(0, len(l), size): 
        
        yield l[i:i + size]

def load_conf():
    
    path = os.path.join(os.getcwd(), 'data', 'conf.yaml')
    
    if not os.path.exists(path):
        raise RuntimeError('Configuratio file `{path}` not found! Please take a look at the README.md')
    
    return OmegaConf.load(path)

def check_exist(path : str):
    msg = None
    if os.path.isdir(path):
        if len(os.listdir(path)) == 0:
            msg = f'Directory `{path}` is emtpy!'
    elif os.path.isfile(path):
        if not os.path.exists(path):
            msg = f'File `{path}` not found!'
            
    if msg is not None:
        raise RuntimeError(f'{msg} Please take a look at the `README.md` file.')



def set_seed(seed : int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(path , *args, **kwargs):
    
    path = glob.os.path.expanduser(path)
    
    with open(str(path)) as infile:
        item = json.load(infile, *args, **kwargs)
    return item


def load_pickle(path, *args, **kwargs):
    """
    Load pickled python object.
    All extra arguments will be passed to `pickle.load`.
    
        args:
        path (str) : system path
    
    return:
        item (object) : python object
    """
    
    path = glob.os.path.expanduser(path)
    
    with open(str(path), mode = "rb") as infile:
        item = pickle.load(infile, *args, **kwargs)
    return item


def natural_sort(items : List[str]) -> List[str]:
    """
    Correctly sort strings in list.  
    See: http://nedbatchelder.com/blog/200712/human_sorting.html
    
    >>> natural_sort(['test10','test1','test12','test2','test22'])
    ['test1', 'test2', 'test10', 'test12', 'test22']
    """

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        def tryint(s):
            try:
                return int(s)
            except ValueError:
                return s
            
        return [ tryint(c) for c in re.split('([0-9]+)', s) ] 
        
    return sorted(items, key = alphanum_key)

def iglob(path : str, ext : Optional[str]  = None, recursive : bool = False, ordered : bool = False) -> Iterable[str]:
    """
    Iterator yielding paths matching a path pattern.
    If the extension is not provided all files are returned. 
    
    Note: if `ordered=True` all paths will be load into memory, otherwise lazy loading is used. 
    
    args:
        path (str) : system path
        ext (str) : file extension
        recursive (bool) : check subfolder
        
    >>> list(iglob(path = '.', ext = 'py', ordered = True))
    ['./__init__.py', './ieaiaio.py', './setup.py']
    """
    
    path = glob.os.path.expanduser(path)
    
    ext = "*.{}".format(ext) if ext is not None else "*"
    
    splits = [path,ext] 
    
    if recursive:
        splits.insert(1,"**")
    
    pregex = glob.os.path.join(*splits)
            
    path_gen = natural_sort(glob.iglob(pregex, recursive = recursive)) if ordered else glob.iglob(pregex, recursive = recursive)
    
    for p in path_gen:
        yield p



def valid_xml_char_ordinal(c : str) -> bool:
    """
    Check if its valida charachter
    """
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
        )

def write_collection(outpath : str, documents : List[bioc.BioCDocument]) -> None:
    
    collection = bioc.BioCCollection()
    collection.documents = documents
    
    with bioc.BioCXMLDocumentWriter(outpath) as writer:
        writer.write_collection_info(collection)
        for document in collection.documents:
            for p in document.passages:
                p.text = ''.join(c if valid_xml_char_ordinal(c) else '?' for c in p.text)
            writer.write_document(document)

def pubtator_api_fetch_annotated_documents(pmids : List[str], outpath : str) -> None:
    """
    Download PubMed abstracts annotated by PubTator model via its API.
    """

    url = 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml'
        
    json_data = {'pmids' : pmids}
    
    r = requests.post(url, json = json_data)
    
    collection = bioc.loads(r.text)
    
    write_collection(documents = collection.documents, outpath = outpath)
    
