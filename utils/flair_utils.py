#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

import itertools
from collections import defaultdict
import torch
from flair.tokenization import SciSpacyTokenizer
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from torch.utils.data import ConcatDataset
from flair.data import Corpus, Dictionary, Sentence, Span
from flair.datasets import ColumnDataset
from typing import List, Optional, Dict


def get_hunflair_tagger(name : str, tag_dictionary : Optional[Dictionary] = None) -> SequenceTagger:
    """
    Instantiate a HunFlair tagger according to name. 
    Get basic tagger wit `base`.
    """
    
    if name == 'base':
        
        embedding_types = [
            WordEmbeddings("pubmed"),
            FlairEmbeddings("pubmed-forward"),
            FlairEmbeddings("pubmed-backward"),
        
        ]
        
        embeddings = StackedEmbeddings(embeddings=embedding_types)
        
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type="ner",
            use_crf=True,
            locked_dropout=0.5
        )
        
    else:
        
        tagger: SequenceTagger = SequenceTagger.load(name)
        
    return tagger

class Metric(object):
    """
    Class to compute Precision, Recall and F1.
    """
    

    def __init__(self, name : str):
        self.name = name

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fp(class_name)), 4)
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(self.get_tp(class_name) / (self.get_tp(class_name) + self.get_fn(class_name)), 4)
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(2 * (self.precision(class_name) * self.recall(class_name))
                         / (self.precision(class_name) + self.recall(class_name)), 4)
        return 0.0

    def accuracy(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) > 0:
            return round(
                (self.get_tp(class_name))
                / (self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)),
                4)
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [self.accuracy(class_name) for class_name in self.get_classes()]

        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)

        return 0.0

    def get_classes(self):
        all_classes = set(itertools.chain(*[list(keys) for keys
                                            in [self._tps.keys(), self._fps.keys(), self._tns.keys(),
                                                self._fns.keys()]]))
        all_classes = [class_name for class_name in all_classes if class_name is not None]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return '{}\t{}\t{}\t{}'.format(
            self.precision(),
            self.recall(),
            self.accuracy(),
            self.micro_avg_f_score(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return '{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE'.format(
                prefix)

        return 'PRECISION\tRECALL\tACCURACY\tF-SCORE'

    @staticmethod
    def to_empty_tsv():
        return '\t_\t_\t_\t_'

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            '{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}'.format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name), self.get_fp(class_name), self.get_fn(class_name), self.get_tn(class_name),
                self.precision(class_name), self.recall(class_name), self.accuracy(class_name),
                self.f_score(class_name))
            for class_name in all_classes]
        return '\n'.join(all_lines)


def span2ann(span : Span) -> Dict:
    
    ann = {'start' : span.start_pos, 
           'end' : span.end_pos, 
           'text' : span.text, 
           'type' : span.tag,
           'score' : round(span.score, 2)}
    
    return ann

def get_sents_anns_from_flair_sent(sent : Sentence):
        
    anns = [span2ann(span = span) for span in sent.get_spans()]
    
    return anns

def get_flair_sents(tagger : SequenceTagger, sents : List[str], tokenizer = SciSpacyTokenizer()):
        
    sents = [Sentence(sent, use_tokenizer=tokenizer) for sent in sents]
        
    tagger.predict(sents, mini_batch_size = len(sents))
    
    anns = [get_sents_anns_from_flair_sent(s) for s in sents]
        
    return anns

def check_accelerator():
    
    if not torch.cuda.is_available():
        logger.warning('There are not GPUs in this machine. This will be considerably slow...')

class MultiFileColumnCorpus(Corpus):
    def __init__(
            self,
            column_format: Dict[int, str],
            train_files=None,
            test_files=None,
            dev_files=None,
            tag_to_bioes=None,
            column_delimiter: str = r"\s+",
            comment_symbol: str = None,
            encoding: str = "utf-8",
            document_separator_token: str = None,
            skip_first_line: bool = False,
            in_memory: bool = True,
            label_name_map: Dict[str, str] = None,
            banned_sentences: List[str] = None,
            *args, **kwargs
    ):
        """
            Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
            :param data_folder: base folder with the task data
            :param column_format: a map specifying the column format
            :param train_files: the name of the train files
            :param test_files: the name of the test files
            :param dev_files: the name of the dev files, if empty, dev data is sampled from train
            :param tag_to_bioes: whether to convert to BIOES tagging scheme
            :param column_delimiter: default is to split on any separatator, but you can overwrite for instance with "\t"
            to split only on tabs
            :param comment_symbol: if set, lines that begin with this symbol are treated as comments
            :param document_separator_token: If provided, sentences that function as document boundaries are so marked
            :param skip_first_line: set to True if your dataset has a header line
            :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
            :param label_name_map: Optionally map tag names to different schema.
            :param banned_sentences: Optionally remove sentences from the corpus. Works only if `in_memory` is true
            :return: a Corpus with annotated train, dev and test data
        """
        # get train data
        train = ConcatDataset([
            ColumnDataset(
                train_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for train_file in train_files
        ]) if train_files and train_files[0] else None

        # read in test file if exists
        test = ConcatDataset([
            ColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for test_file in test_files
        ]) if test_files and test_files[0] else None

        # read in dev file if exists
        dev = ConcatDataset([
            ColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            ) for dev_file in dev_files
        ]) if dev_files and dev_files[0] else None

        super(MultiFileColumnCorpus, self).__init__(train, dev, test, *args, **kwargs)    
    