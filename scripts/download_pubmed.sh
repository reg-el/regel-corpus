#!/bin/bash

#TODO
###########################################################
## EDIT THESE VARIABLES ACCORDING TO YOUR CONFIGURATION!
###########################################################
DIR=./data/pubmed/raw
###########################################################

echo "Downloding files from pubmed/baseline"

START=1
END=1062

for INDEX in $(seq -f "%04g" $START $END); do

    if ! test -f "$DIR/pubmed21n$INDEX.xml"; then
        wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n$INDEX.xml.gz -O $DIR/pubmed21n$INDEX.xml.gz
    fi
    
echo $i; done

echo "Downloding files from pubmed/updatefiles"

START=1063
END=1666

for INDEX in $(seq -f "%04g" $START $END); do

    if ! test -f "$DIR/pubmed21n$INDEX.xml"; then
        wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/pubmed21n$INDEX.xml.gz -O $DIR/pubmed21n$INDEX.xml.gz
    fi
    
echo $i; done
