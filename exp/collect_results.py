#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

###
## NEN
###
#def evaluate(test_ner_file, predictions_file):
    
    ## with open(test_queries_file) as infile:
    ##     test_queries = [q.strip() in q infile.readlines()]
    
    #metric = Metric('Hunflair+BioSyn EL')
    
    #with open(test_ner_file) as infile:
        #test_ner = [q.strip() for q in infile.readlines()]
    
    #with open(predictions_file) as infile:
        #predictions = json.load(infile)
        #test_nen = predictions['queries']
        
        
    ## y_true = [nen_pred['mentions'][0]['golden_cui'] for nen_pred in test_nen]
    ## y_pred = [nen_pred['mentions'][0]['candidates'][0]['cui'] for nen_pred in test_nen]
    ## found = [int(ner_pred.split('||')[-1]) for ner_pred in test_ner]
    
    ## y_true = [y for idx,y in enumerate(y_true) if found[idx] == 1]
    ## y_pred = [y for idx,y in enumerate(y_pred) if found[idx] == 1]
    
    ## p, r, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred = y_pred, average = 'micro')
        
        
    #for ner_pred, nen_pred in zip(test_ner, test_nen):
        
        #ner_found = int(ner_pred.split('||')[-1])
        
        #gold_cui = nen_pred['mentions'][0]['golden_cui']
        
        #top1 = nen_pred['mentions'][0]['candidates'][0]['cui']
                
        #if not ner_found:
            
            #metric.add_fn(gold_cui)
        
        #else:
            
            #if gold_cui == top1:
                
                #metric.add_tp(gold_cui)
            
            #else:
                
                #metric.add_fp(top1)
                
                
    #p = metric.precision()
    #r = metric.recall()
    #f1 = metric.f_score()
                
    #return p, r, f1
            
#if __name__ == '__main__':
    
    #args = parse_args()
    
    #results = {}
    
    #for entity in ['CellLine', 'Disease']: 
        
        #if entity not in results:
            #results[entity] = {'p' : [],
                               #'r' : [],
                               #'f1' : []}

        #for fold in range(5):
            
            ## test_queries_file = os.path.join(args.nen_datasets, entity, f'fold{fold}', 'preprocessed_test', 'test.concept')
            #test_ner_file = os.path.join(args.nen_datasets, entity, f'fold{fold}', 'preprocessed_test', 'test.ner')
            #predictions_file = os.path.join(args.nen_results, entity, f'fold{fold}', 'predictions_eval.json')
            
            #p,r, f1 = evaluate(
                ## test_queries_file = test_queries_file,
                     #test_ner_file = test_ner_file,
                     #predictions_file = predictions_file)
            
            #results[entity]['p'].append(p)
            #results[entity]['r'].append(r)
            #results[entity]['f1'].append(f1)
    
    
    #for e in results:
        
        #p  = round(np.mean(results[e]['p']),2)
        #r  = round(np.mean(results[e]['r']),2)
        #f1  = round(np.mean(results[e]['f1']),2)
        
        #p_std  = round(np.std(results[e]['p']),2)
        #r_std  = round(np.std(results[e]['r']),2)
        #f1_std = round(np.std(results[e]['f1']),2)
        
        #print(f'{e} - P: {p} ({p_std}) - R: {r} ({r_std}) - F1: {f1} ({f1_std})')
    
    
###
# NER
###

# def collect_results(base_out):
    
#     results = {}
    
#     for run_dir in iou.iglob(base_out):
        
#         run_name = iou.fname(run_dir)
        
#         results[run_name] = {}
#         results[run_name]['precision'] = []
#         results[run_name]['recall'] = []
#         results[run_name]['f1'] = []
        
#         for i,fold_dir in enumerate(iou.iglob(run_dir)):
            
#             fold_result_file = os.path.join(fold_dir, 'training.log')
            
#             with open(fold_result_file) as infile:
                
#                 lines = infile.readlines()
                
#                 result_line = lines[-6]
                
#                 result_line = [i for i in result_line.split(' ') if i not in ['','micro','avg']]
                
#                 precision = float(result_line[0])
#                 recall = float(result_line[1])
#                 f1score = float(result_line[2])
    
                
#                 results[run_name]['precision'].append(precision)
#                 results[run_name]['recall'].append(recall)
#                 results[run_name]['f1'].append(f1score)
                
                
#     avg_results(results)
    

# def avg_results(results):
    
    
#     df = {}
    
#     for run, run_results in results.items():
        
#         # print(f"Run: {run}")
        
#         p = run_results.get('precision')
#         r = run_results.get('recall')
#         f1 = run_results.get('f1')
        
#         p_avg, p_std = np.round(np.mean(p), 2), np.round(np.std(p), 2)
#         r_avg, r_std = np.round(np.mean(r), 2), np.round(np.std(r), 2)
#         f1_avg, f1_std = np.round(np.mean(f1), 2), np.round(np.std(f1), 2)
        
#         df[run.upper()] = {}
#         df[run.upper()]['Precision'] = f'{p_avg} ($ pm {p_std}$)'
#         df[run.upper()]['Recall'] = f'{r_avg} ($ pm {r_std}$)'
#         df[run.upper()]['F1'] = f'{f1_avg} ($ pm {f1_std}$)'
    
    
#     df = pd.DataFrame.from_dict(df, orient = 'index')
    
#     print(df.to_latex())
