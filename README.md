# RegEl Corpus

Scripts to reproduce experiments reported in:


## Requirements


## Configuration file and resources

You just need to create a  configuration file `data/conf.yaml` according to where you want to store all resources. An example configuration is `data/conf-example.yaml` which stores everything in the repository 'data' directory.

Results of all experiments will be placed in the `results` directory.

A preprocessed version of the corpus is already available in `data/corpus`.

### NEN

To run the NEN experiments you need to download the following resources updating the configuration file accordingly:

- Brenda Tissue Ontology from [here](https://github.com/BRENDA-Enzymes/BTO/blob/master/bto-full.obo) (`<ontology>/tissue` in `data/conf.yaml`)
- MONDO Disease Ontology from  [here](purl.obolibrary.org/obo/mondo.obo) (`<ontology>/disease` in `data/conf.yaml`)

### COO

*WARNING*: Make sure you have at least 200GB available.

To run the COO experiments you need to download and extract the following resources updating the configuration file accordingly:

* PubTator Central annotations for:
    - gene [here](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/gene2pubtatorcentral.gz) (extract archive and place inside `<pubtator>/<annotations>` ) 
    - disease [here](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/disease2pubtatorcentral.gz) (extract archive and place inside `<pubtator>/<annotations>`)
    
* PubMed abstracts from [here](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) (`<pubmed>` in `data/conf.yaml`):
    - you can use the script `scripts/download_pubmed.sh` (modifying the output directory to reflect the one in `data/conf.yaml`)


## Named Entity Recognition

### KFOLD

```python
python -m exp.ner.kfold
```

This will run all HunFlair models with 5-fold evaluation protocol.

```python
python -m exp.ner.get_kfold_results
```

Results (model training + predictions) will be printed out and stored in `<results dir>/ner/kfold`

### ZERO-SHOT

```python
python -m exp.ner.zero_shot
```
Results will be printed out and stored in `<results dir>/ner/zero_shot.tsv`

## Named Entity Normalization

### KFOLD

*WARNING*: To run this experiment you first need to run the [NER one](#Named Entity Recognition).

This experiment makes use of BioSyn [1](#References) so you first need to convert the ontologies to the required format:

```python
python -m preprocessing.obo2biosyn
```

You can now clone the official BioSyn repository:

```bash
$git clone https://github.com/dmis-lab/BioSyn
```

Modify the script `scripts/run_biosyn_kfold.sh` according to you configuration file. Move it inside the cloned repository grant it execution permission:

```bash 
$ chmod +x ./run_biosyn_kfold.sh
```

You can now run the 5-fold evaluation with BioSyn.


```python
python -m exp.nen.get_kfold_results
```

Results (model training + predictions) are store in `<results dir>/nen/kfold`

### ZERO-SHOT

```python
pyton -m exp.nen.zero_shot
```

Results will be printed out and stored in `<results dir>/nen/zero_shot.tsv`

## CO-occurence analysis

To regenerate the co-occurences you first need to train the HunFlair models on the entire corpus:

```python
python -m exp.coo.train_hunflair
```

Secondly you need to preprocess the PubTator annotations:

```python 
python -m exp.preprocessing.parse_pubtator_annotations
```

You also need to download all PubMed abstracts:

```bash
$ chmod +x ./scripts/download_pubmed.sh
$ ./scripts/download_pubmed.sh
```

and build a sqlite database for fast retrival:

```python
python -m exp.coo.build_pubmed_db --workers <available_cores> --overwrite <if_something_went_wrong>
```

Once all these resources are correctlyb configured you can run the annotation with:

```python
CUDA_VISIBLE_DEVICES=0 python -m exp.coo.annotate --batch_size 64 --idxs 1,24 --range
```

This will retrieve abstracts from `<pubmed dir>/pubmed21.db` and annotate them with the trained HunFlair models according to the pmids specified in the files contained in `<pubmed dir>/<shards dir>`. 

Note that the scripts runs on a single GPU (the one with index 0 if not `CUDA_VISIBLE_DEVICES` is not specified). If you have more you can tweak the script

```bash
$ chmod +x ./scripts/run_annotate.sh
$ ./scripts/run_annotate.sh
```
Results will be stored in `<results dir>/coo/raw`.


## References

1. Sung, M., Jeon, H., Lee, J., & Kang, J. (2020). Biomedical entity representations with synonym marginalization. arXiv preprint arXiv:2005.00239.
