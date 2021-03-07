# GrailQA: Strongly <ins>G</ins>ene<ins>ra</ins>l<ins>i</ins>zab<ins>l</ins>e Question Answering
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
<img width="1175" alt="image" src="https://user-images.githubusercontent.com/15921425/110228546-f2193380-7ecf-11eb-8cbd-c5097a064ee4.png">

>GrailQA is a new large-scale, high-quality KBQA dataset with 64,331 questions annotated with both answers and corresponding logical forms in different syntax (i.e., SPARQL, S-expression, etc.). It can be used to test three levels of generalization in KBQA: **i.i.d.**, **compositional**, and **zero-shot**.

>For dataset and leaderboard, please refer to the [homepage of GrailQA](https://dki-lab.github.io/GrailQA/). In this repo, we help you to reproduce the results of our baseline models and to train new models using our code.

## Overview
To study the three levels of generalization in KBQA, we implement a line of baseline models of different natures, namely, **Transduction+BERT**, **Transduction+GloVe**, **Ranking+BERT**, and **Ranking+GloVe**. Our implementation is based on [AllenNLP](https://github.com/allenai/allennlp). 
### Package Description
```
GrailQA/
├─ model_configs/
    ├─ train/: Configuration files for training
    ├─ test/: Configuration files for inference
├─ data/: Data files for training, validation, and test
├─ ontology/: Processed Freebase ontology files
    ├─ domain_dict: Mapping from a domain in Freebase Commons to all schema items in it
    ├─ domain_info: Mapping from a schema item to a Freebase Commons domain it belongs to
    ├─ fb_roles: Domain and range information for a relation (Note that here domain means a different thing from domains in Freebase Commons)
    ├─ fb_types: Class hierarchy in Freebase
    ├─ reverse_properties: Reverse properties in Freebase 
├─ bert_configs/: BERT configuration used by pytorch_transformer, which you are very unlikely to modify
├─ entity_linking/: Entity linking results 
├─ vocabulary/: Preprocessed vocabulary, which is only required by our GloVe-based models
├─ cache/: Cached results for SPARQL queries, which are used to accelerate the experiments by caching many SPARQL query results offline
├─ saved_models/: Trained models
├─ utils/
    
├─ 
├─
├─
├─
├─
```

## Setup
There are several steps you need to do before running our code.
1. Follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to run your own Virtuoso service. After starting your virtuoso service, replace the url in `utils/sparql_executer.py` with your own address.
2. Download cache files from https://1drv.ms/u/s!AuJiG47gLqTznjfRRxdW5YDYFt3o?e=GawH1f and put all the files under `cache/`.
3. Download trained models from https://1drv.ms/u/s!AuJiG47gLqTznjaviBVyXM4tOa4J?e=XaGp8d and put all the files under `saved_models/`.
4. Download GrailQA dataset and put it under `data/`.
5. Install all required libraries:
```
$ pip install -r requirements.txt
```
**(Note: you do not need to install AllenNLP by yourself, because we have included our local version of AllenNLP in this repo.)**

## Reproduce Our Results
The predictions of our baseline models can be found via [CodaLab](https://worksheets.codalab.org/worksheets/0x53f31035f34e4b6194ebe16179944297).
Run `predict` command to reproduce the predictions. There are several commands to configure to run `predict`:
```
python run.py predict
  [path_to_saved_model]
  [path_to_test_data]
  -c [path_to_the_config_file]
  --output-file [results_file_name] 
  --cuda-device [cuda_device_to_use]
```
Specifically, to run Ranking+BERT:
```
PYTHONHASHSEED=23 python run.py predict saved_models/BERT/model.tar.gz data/grailqa_v1.0_test_public.json --include-package bert_constrained_seq2seq --include-package bert_seq2seq_reader --include-package utils.bert_interface --use-dataset-reader --predictor seq2seq -c model_configs/test/bert_ranking.jsonnet --output-file bert_ranking.txt --cuda-device 0
```
To run Ranking+GloVe:
```
PYTHONHASHSEED=23 python run.py predict predict saved_models/GloVe/model.tar.gz data/grailqa_v1.0_test_public.json --include-package constrained_seq2seq --include-package constrained_seq2seq_reader --predictor seq2seq --use-dataset-reader -c model_configs/test/glove_ranking.jsonnet --output-file glove_ranking.txt --cuda-device 0
```
To run Transduction+BERT:
```
PYTHONHASHSEED=23 python run.py predict saved_models/BERT/model.tar.gz data/grailqa_v1.0_test_public.json --include-package bert_constrained_seq2seq --include-package bert_seq2seq_reader --include-package utils.bert_interface --use-dataset-reader --predictor seq2seq -c model_configs/test/bert_vp.jsonnet --output-file bert_vp.txt --cuda-device 0
```
To run Transduction+GloVe:
```
PYTHONHASHSEED=23 python run.py predict predict saved_models/GloVe/model.tar.gz data/grailqa_v1.0_test_public.json --include-package constrained_seq2seq --include-package constrained_seq2seq_reader --predictor seq2seq --use-dataset-reader -c model_configs/test/glove_vp.jsonnet --output-file glove_vp.txt --cuda-device 0
```

### Entity Linking
We also provide instructions on reproduce our entity linking results to benefit future research. Similar to most existing KBQA methods, entity linking is a separate module from our main model. If you just want to run our main models, you do not need to re-run our entity linking module because our models directly retrieve the produce entity linking results under `entity_linking/`.
(To be continued...)


## Train New Models
You can also use our code to train new models.
### Training Configuration
### Training Command

## Citation
```
@article{gu2021beyond,
  title={Beyond IID: three levels of generalization for question answering on knowledge bases},
  author={Gu, Yu and Kase, Sue and Vanni, Michelle and Sadler, Brian and Liang, Percy and Yan, Xifeng and Su, Yu},
  journal={The World Wide Web Conference},
  year={2021},
  organization={ACM}
}
```
