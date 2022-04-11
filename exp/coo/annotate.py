#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

import os
import argparse
import flair
import sqlite3
from itertools import islice
import spacy
import json
from sentence_splitter import SentenceSplitter
from typing import List, Tuple, Any, Iterable
import utils

SENTENCE_SPLITTER = SentenceSplitter(language='en')
SCISPACY_MODEL = spacy.load("en_core_sci_sm", disable = ["ner"])

class SentenceTokenizer:
    
    @staticmethod
    def tokenize(text : str) -> List[str]:
        
        text = SCISPACY_MODEL(text)
        
        sentences = [str(s) for s in text.sents]
        
        sentences = [SENTENCE_SPLITTER.split(s) for s in sentences]
        
        sentences = [s for sublist in sentences for s in sublist]
        
        return sentences

def parse_args():
    
    parser = argparse.ArgumentParser(description='Annotate PubMed abstracts with HunFlair models for regulatory elements',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=64, type = int,  help = 'Mini-batch size (# of sentences to process)')
    parser.add_argument('--idxs', required = True, nargs='+', type=int,  help = 'Comma separate file indices, e.g. 1,2,3')
    parser.add_argument('--range', action = 'store_true',  help = 'Shards indices is a range, e.g. `1,3` = 1,2,3')
    
    args = parser.parse_args()
    
    if args.range:
        start = args.idxs[0]
        end = args.idxs[1]
        args.idxs = list(range(start, end+1))
        
    if len(args.idxs) > 50:
        raise ValueError('Cannot process `{len(args.idxs)}` > 50 files at a time!')
    
    return args


        
def chunkize_generator(g : Iterable[Any], size : int = 128) -> Iterable[Any]:
    
    """Efficiently split `gen` into chunks of size `k`.
    
        Args:
            gen: Iterator to chunk.
            k: Number of elements per chunk.
        
        Yields:
            Chunks as a list.
    """ 
    while True:
        chunk = [*islice(g, 0, size)]
        if chunk:
            yield chunk
        else:
            break
    
    

def get_sents_generator(files_dir : str, 
                        pubmed_db : str, 
                        idxs : List[int]) -> Iterable[Tuple[int ,str, str]]:
    
    connection = sqlite3.connect(pubmed_db)
    
    cursor = connection.cursor()
    
    batch_size = 10000
    
    missing = set()
    
    for idx in idxs:
        
        shard_path = os.path.join(files_dir, f'shard{idx}.json')
        with open(shard_path) as infile:
            pmid2cuis = json.load(infile)
        
        pmids = [str(pmid) for pmid, in list(pmid2cuis.keys())]
        
        for batch_pmids in utils.chunkize_list(pmids, batch_size):
            
            batch_pmids = tuple(list(batch_pmids))
        
            rows = cursor.execute(f"SELECT pmid, title, abstract FROM pubmed21 WHERE pmid IN {batch_pmids};")
            
            rows = rows.fetchall()
            
            rows = [(pmid, [title, abstract]) for (pmid, title, abstract) in rows]
        
            pmid2text = {pmid : ' '.join([t for t in texts if t is not None]) for pmid, texts in rows}
            
            missing.update([i for i in batch_pmids if i not in pmid2text])
            
            logger.info(f"Fetched {len(pmid2text)} documents: start generating sentences...")
            
            for pmid, text in pmid2text.items():
                
                sents = SentenceTokenizer.tokenize(text)
                
                for sent in sents:
                        
                    yield (idx, pmid, sent)
        
        missing_dir = os.path.join(args.out, 'missing')
        os.makedirs(missing_dir, exist_ok=True)
        
        with open(os.path.join(missing_dir, f'./missing-{idx}.pmids'), 'w') as outfile:
            for m in missing:
                outfile.write(f'{m}\n')
            
    logger.info(f"Completed iterating over shards {idx}")
    connection.close()
    
def hunflair_annotate(files_dir : str, 
                      pubmed_db : str, 
                      idxs : List[int], 
                      taggers : List[str], 
                      outdir : str,
                      batch_size : int = 128):
    
    gpu_idx = os.environ['CUDA_VISIBLE_DEVICES']
    logger.info(f"#GPU:{gpu_idx} - Start annotation with shards `{args.shards_idxs}`")

    logger.info("Loading models...")
    tagger = flair.models.MultiTagger.load(taggers)
    tokenizer = flair.tokenization.SciSpacyTokenizer()
    
    sents_generator = get_sents_generator(files_dir = files_dir,
                                          pubmed_db = pubmed_db,
                                          idxs= idxs)
    
    tot_sents = 0
    kept_sents = 0  
    curr_pmid = ''
    sent_idx = 0
    file_handles = {}
    pmids_seen = set()
            
    for elements in chunkize_generator(sents_generator, size = batch_size):
        
        idxs, pmids, sents = zip(*elements)
        
        pmids_seen.update(pmids)
    
        tot_sents += len(sents)
        
        if (tot_sents%(args.batch_size*100))==0:
            for shardidx, handles in file_handles.items():
                for k, handle in handles.items():
                    handle.flush()
            logger.info(f"#PROGRESS - parsed {tot_sents} sentences (~{len(pmids_seen)} documents): {kept_sents} with annotations")
        
        sents_flair_anns = utils.get_flair_sents(tagger = tagger, 
                                                 tokenizer = tokenizer, 
                                                 sents = sents)
        
        for idx, pmid, sent, anns in zip(idxs,
                                         pmids, 
                                         sents,
                                         sents_flair_anns):
            if len(anns) > 0:
                
                kept_sents += 1
                
                if idx not in file_handles:
                    file_handles[idx] = {}
                    file_handles[idx]['sents'] = open(os.path.join(outdir, f'sents-{idx}.txt'), 'w')
                    file_handles[idx]['sents'].write('PMID\tSID\tSENTENCE\n')
                    file_handles[idx]['assoc'] = open(os.path.join(outdir, f'assoc-{idx}.txt'), 'w')
                    file_handles[idx]['assoc'].write('PMID\tSID\tRE-TYPE\tRE-TEXT\tRE-SCORE\n')
                
                if curr_pmid == '':
                    curr_pmid = pmid
                
                elif pmid == curr_pmid:
                    sent_idx += 1
                    
                elif curr_pmid != pmid:
                    curr_pmid = pmid
                    sent_idx = 0
                
    
                file_handles[idx]['sents'].write(f'{pmid}\t{sent_idx}\t{sent}\n')
                
                for a in anns:
                    atype = a.get('type').upper()
                    atext = a.get('text')
                    ascore = a.get('score')
                    
                    file_handles[idx]['assoc'].write(f'{pmid}\t{sent_idx}\t{atype}\t{atext}\t{ascore}\n') 
                        
    logger.info(f"Completed processing: written to file {kept_sents} out of {tot_sents} parsed sentences")
    
    for shardidx, handles in file_handles.items():
        for k, handle in handles.items():
            handle.close()
            
if __name__ == '__main__':
    
    logging.getLogger('flair').setLevel(logging.ERROR)
    
    args = parse_args()
    
    conf = utils.load_conf()
        
    os.makedirs(args.out, exist_ok=True)
    
    taggers = []
    
    for entity in ['enhancer', 'promoter', 'tfbs']:
        
        tagger = os.path.join(conf['results'],  'models', 'full_corpus', entity)
        
        tagger.append(tagger)
    
    outdir = os.path.join(conf['results'], 'coo', 'raw')
    os.makedirs(outdir, exist_ok=True)
    
    pubmed_db = os.path.join(conf['pubmed'], 'pubmed21.db')
    files_dir = os.path.join(conf['pubtator']['annotations'], 'shards')
    
    hunflair_annotate(files_dir = files_dir,
                      pubmed_db = pubmed_db,
                      outdir = outdir,
                      taggers = taggers,
                      idxs = args.idxs,
                      batch_size = args.batch_size)

        
