#!/bin/bash


PYTHON=$(which python)
echo "Using python: $PYTHON" 

#TODO
###########################################################
## EDIT THESE VARIABLES ACCORDING TO YOUR CONFIGURATION!
###########################################################
GPU=0
DICT_DIR=./data/ontology/
DATA_DIR=./data/corpus/biosyn
OUTPUT_DIR=./data/results/nen/kfold
###########################################################


MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
ENTITIES="tissue disease"
FOLDS=4

for ENTITY in $ENTITIES; do
    
    
    for FOLD in $(seq 0 $FOLDS); do
    
        
    echo "Start training BioSyn model - $ENTITY - fold $FOLD"
        
        
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
                
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
               
    --train_dictionary_path ${DICT_DIR}/${ENTITY}/train_dictionary.txt \
                
    --train_dir ${DATA_DIR}/${ENTITY}/fold${FOLD}/preprocessed_train \
                
    --output_dir ${OUTPUT_DIR}/models/${ENTITY}/fold${FOLD} \
                
    --use_cuda \
                
    --topk 20 \
                
    --epoch 10 \
                
    --train_batch_size 16\
                
    --learning_rate 1e-5 \
                
    --max_length 25 \
                
    
        
    echo "Evaluate BioSyn model - $ENTITY - fold $FOLD"
                
    
        
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
                
    --model_name_or_path ${OUTPUT_DIR}/models/${ENTITY}/fold${FOLD} \
                
    --dictionary_path ${DICT_DIR}/${ENTITY}/train_dictionary.txt \
                
    --data_dir ${DATA_DIR}/${ENTITY}/fold${FOLD}/preprocessed_test \
                
    --output_dir ${OUTPUT_DIR}/predictions/${ENTITY}/fold${FOLD} \
                
    --use_cuda \
                
    --topk 20 \
                
    --max_length 25 \
                
    --save_predictions
                
        
    echo "Delete BioSyn model - $ENTITY - fold $FOLD"
        
    rm -rf ${OUTPUT_DIR}/models/${ENTITY}/fold${FOLD}
    
    
    
    done
    
    
    
    echo "Completed $ENTITY : fold $FOLD"


done
echo "Completed running NEN experiment"

    
