#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)


import os
import re
import bioc
import copy
import numpy as np
from sentence_splitter import SentenceSplitter
from typing import List, Tuple, Any
import utils

class Bioc2Conll(object):
    def __init__(self):
        self._sentinel_token = '@'
        self._sentence_splitter = SentenceSplitter(language='en')
        self._tokenization_regexp = re.compile('([0-9a-zA-Z]+|[^0-9a-zA-Z])')
        self.ENTITY_TYPES = ['Enhancer','Promoter',
                            'Tfbs','Gene','Tissue','Disease','Species','Variant']
    
    def get_sents_anns(self, text : str,
                       anns : List[bioc.BioCAnnotation], 
                       did : str) -> Tuple[List[str], List[List[bioc.BioCAnnotation]]]:
        
        text = text.replace(self._sentinel_token, ' ')
        
        anns = sorted(anns, key = lambda x : x.total_span.offset)
                
        text = self._anns2sentinels(text = text, anns = anns, did = did)
        
        sents = self._sentence_splitter.split(text)
        
        sents_anns = self._match_anns2sents(sents = sents, anns = anns, did = did)
        
        return sents_anns

    
    def _anns2sentinels(self, text : str,
                        anns : List[bioc.BioCAnnotation],
                        did : str) -> str:
        """
        Given a span of text defined by offset, replace its characters with a sentinel token
        """
        
        chars = np.asarray(list(text))
        
        for a in anns:
            
            start = a.total_span.offset
            end = a.total_span.end
            ann_text = a.text
            length = len(ann_text)
            
            sentinel_tokens = list(self._sentinel_token*length) 
            
            try:
                chars[start:end] = sentinel_tokens
            except ValueError:
                try:
                    end = start + length
                    chars[start:end] = sentinel_tokens
                except ValueError:
                    # print('cannot match annotation: `{}`'.format(ann_text))
                    print(text)
                    print([a.text for a in anns])
                    raise ValueError('cannot match annotation: `{}`'.format(ann_text))
    
        text_with_sentinel = "".join(chars)
        
        return text_with_sentinel


    def _match_anns2sents(self, sents : List[str],
                          anns : List[bioc.BioCAnnotation], 
                          did : str) -> Tuple[List[str], List[List[bioc.BioCAnnotation]]]:
        
        sents_anns = [["",[]] for sent in sents]
        
        sents_matched = [False]*len(sents)
        anns_matched = [False]*len(anns)
        
        # only 'look ahead' regexp match
        last_match_in_s = 0
        
        for sent_idx,s in enumerate(sents):
    
            # skip sentence if already processed
            if sents_matched[sent_idx]:
                continue
            
            for ann_idx,a in enumerate(anns):
                
                # skip annotation if already used
                if anns_matched[ann_idx]:
                    continue
                
                length = len(a.text)
                
                pattern = f"{self._sentinel_token}{{{length}}}" #re.escape()
                
                pattern = re.compile(pattern)
                
                # check for exact match of # of sentinel tokens
                # from last match found in sentence
                match = re.search(pattern,s[last_match_in_s:])
                
                if match is not None:
                    
                    last_match_in_s = match.start()
                    
                    # # replace sentinel_token with original text
                    sentinel_chars = self._sentinel_token*length
                    
                    s = re.sub(sentinel_chars, a.text, s, 1)
                                                            
                    sents_anns[sent_idx][0] = s
                    sents_anns[sent_idx][1].append(a)
                    
                    anns_matched[ann_idx] = True
            
            sents_anns[sent_idx][0] = s
            sents_matched[sent_idx] = True
            
            # new sentence, start looking from beginning
            last_match_in_s = 0
        
    
        # if did == '26817450':
        #     print([a.text for a in nested_anns])
        #     for s,anns in sents_anns:
        #         print(s, [a.text for a in anns])
        #         print('\n')        
                
        if not all(anns_matched):
            for s,anns in sents_anns:
                print(s, [a.text for a in anns])
    
            raise ValueError(f"Not all annotations were matched : {did}")
    
        return sents_anns


    def tokenize(self, s : str) -> List[str]:
        
        tokens = [t for t in self._tokenization_regexp.split(s) if t and not t.isspace()]
        
        return tokens

    def get_labeled_anns(self, anns : List[bioc.BioCAnnotation]) -> List[str]:
        
        labeled_anns = []
            
        for a in anns:
            labeled_ann = []
            entity = a.infons.get('type')
            ann_tokens = self.tokenize(a.text)
            if len(ann_tokens) == 1:
                label = f'S-{entity}'
                t = ann_tokens[0]
                labeled_ann.append((t,label))
            else:
                for idx,t in enumerate(ann_tokens):
                    if idx == 0:
                        label = f'B-{entity}'
                    elif idx == len(ann_tokens)-1:
                        label = f'E-{entity}'
                    else:
                        label = f'I-{entity}'
                    labeled_ann.append((t,label))
            labeled_anns.append(labeled_ann)
        
        return labeled_anns
    
    
    def get_sublist_indices(self, sl : List[Any], l : List[Any]) -> List[int]:
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll-1


    def get_anns_indices(self, 
                         did : str, 
                         anns_labeled : List[Tuple[str, str]], 
                         sent_tokens : List[str]) -> List[Tuple[int, int]]:
        
        indices = []
        
        # print("*"*80)
        # print(did)
        
        # print([[tl[0] for tl in al] for al in anns_labeled])
        last_found = 0
        offset = 0
        for al in anns_labeled:
            ann_tokens = [tl[0] for tl in al]
            
            tmp_sent_tokens = sent_tokens[last_found:]
            
            # print(f"Looking into: {tmp_sent_tokens}")
            
            
            idxs = self.get_sublist_indices(ann_tokens, tmp_sent_tokens)
            
            if idxs is None:
                raise ValueError("Likely there is a mismatch between annotation offsets and actual annotations!")
            
            start = idxs[0] + offset
    
            end = idxs[1] + offset
            
            indices.append((start, end))
            
            last_found = end + 1
            offset = len(sent_tokens[:last_found])
            
            
        return indices
        
        
    
    def get_labeled_sent(self, did : str, 
                         sent : str, 
                         anns : List[bioc.BioCAnnotation]) -> List[Tuple[str, str]]:
        
        tokens = self.tokenize(sent)
        
        anns_labeled = self.get_labeled_anns(anns)
        
        #TODO find sublist indices
        # must find all of them (same mention multiple times in one sentence)
        # now it breaks because of partial matching
        # allow only lookahead search from last found
        anns_tokens_idxs = self.get_anns_indices(did, anns_labeled, tokens)
        if len(anns_labeled) !=  len(anns_tokens_idxs):
            # print(tokens)
            # print(anns_labeled)
            # print(anns_tokens_idxs)
            raise ValueError(f"# annotations {len(anns_labeled)} != # annotations indices {len(anns_tokens_idxs)}")
            
        tokens_labeled = [(t,'O') for t in tokens]
        
        for idx,ann_tokens_idxs in enumerate(anns_tokens_idxs):
            start = ann_tokens_idxs[0]
            end = ann_tokens_idxs[1]
            tokens_labeled[start:end+1] = anns_labeled[idx]
            
        return tokens_labeled
    
    def _write_entity_conll_corpus(self, collection : bioc.BioCCollection, outdir : str, entity_type : str):
        
        for d in collection.documents:
            
            filepath = os.path.join(outdir, f'{d.id}.conll')
            
            with open(filepath, 'w') as outfile:
                for p in d.passages:
                    p.annotations = [a for a in p.annotations if a.infons.get('type') == entity_type.upper()]
                    
                    sents_anns = self.get_sents_anns(did = d.id, text = p.text, anns = p.annotations)
                    
                    for idx, (sent, anns) in enumerate(sents_anns):
                        try:
                            labeled_sent = self.get_labeled_sent(sent = sent, anns = anns, did = d.id)
                            for token, label in labeled_sent:
                                outfile.write(f'{token}\t{label}\n')
                        except ValueError:
                            logger.warning(f'Skipping sentence `{sent}` with annotations `{[a.text for a in anns]}`')
                            
                        
                        
                        if idx != len(sents_anns)-1:
                            outfile.write('\n\n')
        logger.info(f'Completed writing corpus for `{entity_type}` entity to `{outdir}`')
    
    def create_conll_corpus(self, bioc_corpus_path : str, outpath : str):
        
        logger.info(f'Start converting in CONLL format BioC file in `{bioc_corpus_path}`')
        
        collection = bioc.load(bioc_corpus_path)
        
        for entity_type in self.ENTITY_TYPES:
            
            outdir = os.path.join(outpath , f'{entity_type}')
            os.makedirs(outdir, exist_ok=True)
            
            by_type_collection = copy.deepcopy(collection)
            
            self._write_entity_conll_corpus(collection=by_type_collection, outdir = outdir, entity_type = entity_type)


def _map_variant_types(c : bioc.BioCCollection) -> bioc.BioCCollection:
    
    for d in c.documents:
        for p in d.passages:
            p.annotations = [a for a in p.annotations if a.infons.get('type') in ['DNAMutation', 'ProteinMutation', 'SNP']]
            for a in p.annotations:
                a.infons['type'] = 'VARIANT' 
    return c


def _remap_offsets(text : str, annotations : List[bioc.BioCAnnotation]) -> bioc.BioCAnnotation:
    
    annotations = sorted(annotations, key = lambda x : x.total_span.offset)
    
    remapped_annotations = []
    
    last_match = 0 
    
    for a in annotations:
        
        pattern = re.compile(re.escape(a.text))
        
        match = re.search(pattern, text[last_match:])
            
        if match is None:
            raise RuntimeError('Failed to remap offsets with marked annotations')
            
        last_match_offset = len(text[:last_match])
        
        last_match = match.end() + last_match_offset
        
        rma = bioc.BioCAnnotation()
        rma.text = a.text
        location = bioc.BioCLocation(offset = match.start() + last_match_offset, length = len(rma.text))
        rma.add_location(location)
        rma.infons['type'] = a.infons.get('type')
        remapped_annotations.append(rma) 
        
    return remapped_annotations


def _remap_collection_offsets(c : bioc.BioCCollection) -> bioc.BioCCollection:
    """
    Offsets from PubTator may differ from our text due to special charachter handling in XML files.
    """
    
    for d in c.documents:
        for p in d.passages:    
            p.annotations = _remap_offsets(text = p.text, annotations = p.annotations)
    
    return c
            

def get_pubtator_collection(ref_c : bioc.BioCCollection, c : bioc.BioCCollection) -> bioc.BioCCollection:
    
    c = _map_variant_types(c = c)
            
    docs = sorted(c.documents, key = lambda x : x.id)
    ref_docs = sorted(ref_c.documents, key = lambda x : x.id)
    
    mapped_docs = []
    for i in range(len(ref_docs)):
        d = ref_docs[i]
        d.passages[0].annotations = [a for p in docs[i].passages for a in p.annotations]
        mapped_docs.append(d)
    
    mapped_c = bioc.BioCCollection()
    mapped_c.documents = mapped_docs
    mapped_c = _remap_collection_offsets(c = mapped_c)
    
    return mapped_c

if __name__ == '__main__':
    
    conf = utils.load_conf()
    
    bioc2conll = Bioc2Conll()
    
    bioc_file = conf['corpus']['bioc']
    conll_dir = conf['corpus']['conll']
    
    if not os.path.exists(conll_dir):
        bioc2conll.create_conll_corpus(bioc_corpus_path = bioc_file,
                                       outpath = conll_dir)
    
    pubtator_bioc_file = conf['pubtator']['corpus_bioc']
    pubtator_conll_dir = conf['pubtator']['corpus_conll']
    os.makedirs(pubtator_conll_dir, exist_ok=True)
        
    corpus = bioc.load(bioc_file)
    pubtator_corpus = bioc.load(pubtator_bioc_file)
    pubtator_corpus =  get_pubtator_collection(ref_c = corpus, c = pubtator_corpus)  
    bioc2conll._write_entity_conll_corpus(collection=pubtator_corpus, outdir = pubtator_conll_dir, entity_type='Variant')
                    