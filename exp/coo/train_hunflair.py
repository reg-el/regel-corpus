#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

import os
from flair.data import Corpus
from flair.trainers import ModelTrainer
import utils
from utils import MultiFileColumnCorpus

if __name__ == '__main__':
    
    conf = utils.load_conf()
    
    logging.getLogger('gensim').setLevel(logging.ERROR)
    
    utils.set_seed(42)
    utils.check_accelerator()
 
    entity2model = {'enhancer' : 'base',
                    'promoter' : 'base',
                    'tfbs' : 'base'}

    column_format = {0: 'text', 1: 'ner'}
    
    corpus_conll_path = conf['corpus']['conll']
    result_dir = os.path.join(conf['results'], 'models', 'full_corpus')
    
    for entity, model_name in entity2model.keys():
        
        run_dir = os.path.join(corpus_conll_path, entity)
        
        corpus : Corpus = MultiFileColumnCorpus(run_dir, 
                                                train_files = list(utils.iglob(run_dir, ext = 'conll')),
                                                column_format = column_format)
                                        
        tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
            
        tagger = utils.get_hunflair_tagger(name = model_name, tag_dictionary = tag_dictionary)
    
        trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
        base_path = os.path.join(result_dir, entity)
        trainer.train(base_path = base_path,
                      train_with_dev=True,
                      train_with_test=True,
                      max_epochs=100,
                      learning_rate=0.1,
                      mini_batch_size=32,
                      checkpoint = True,
                      save_final_model = True)