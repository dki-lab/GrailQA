# GrailQA: Strongly <ins>G</ins>ene<ins>ra</ins>l<ins>i</ins>zab<ins>l</ins>e Question Answering
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/dki-lab/GrailQA/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
[![paper](https://img.shields.io/badge/Paper-WWW-lightgrey?style=flat-square)](https://arxiv.org/abs/2011.07743)
<img width="1175" alt="image" src="https://user-images.githubusercontent.com/15921425/110228546-f2193380-7ecf-11eb-8cbd-c5097a064ee4.png">

>GrailQA is a new large-scale, high-quality KBQA dataset with 64,331 questions annotated with both answers and corresponding logical forms in different syntax (i.e., SPARQL, S-expression, etc.). It can be used to test three levels of generalization in KBQA: **i.i.d.**, **compositional**, and **zero-shot**.

>For dataset and leaderboard, please refer to the [homepage of GrailQA](https://dki-lab.github.io/GrailQA/). In this repository, we help you to reproduce the results of our baseline models and to train new models using our code.

## Package Description
The following lengthy descriptions might seem intimidating. You can view it as a reference to help you understand our code when you try to implement your own model using our implementations later, but for now it is totally fine to skip them if you only want to run our models. :blush:
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
├─ utils/:
    ├─ bert_interface.py: Interface to BERT 
    ├─ logic_form_util: Tools related to logical forms, including the exact match checker for two logical forms
    ├─ search_over_graphs.py: Generate candidate logical forms for our Ranking models
    ├─ sparql_executor: Sparql-related tools
├─ bert_constrained_seq2seq.py: BERT-based model for both Ranking and Transduction
├─ bert_seq2seq_reader.py: Data reader for BERT-based models
├─ constrained_seq2seq.py: GloVe-based model for both Ranking and Transduction
├─ constrained_seq2seq_reader.py: Data reader for GloVe-based models
├─ run.py: Main function
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
Run `predict` command to reproduce the predictions. There are several arguments to configure to run `predict`:
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
$ PYTHONHASHSEED=23 python run.py predict saved_models/BERT/model.tar.gz data/grailqa_v1.0_test_public.json --include-package bert_constrained_seq2seq --include-package bert_seq2seq_reader --include-package utils.bert_interface --use-dataset-reader --predictor seq2seq -c model_configs/test/bert_ranking.jsonnet --output-file bert_ranking.txt --cuda-device 0
```
To run Ranking+GloVe:
```
$ PYTHONHASHSEED=23 python run.py predict predict saved_models/GloVe/model.tar.gz data/grailqa_v1.0_test_public.json --include-package constrained_seq2seq --include-package constrained_seq2seq_reader --predictor seq2seq --use-dataset-reader -c model_configs/test/glove_ranking.jsonnet --output-file glove_ranking.txt --cuda-device 0
```
To run Transduction+BERT:
```
$ PYTHONHASHSEED=23 python run.py predict saved_models/BERT/model.tar.gz data/grailqa_v1.0_test_public.json --include-package bert_constrained_seq2seq --include-package bert_seq2seq_reader --include-package utils.bert_interface --use-dataset-reader --predictor seq2seq -c model_configs/test/bert_vp.jsonnet --output-file bert_vp.txt --cuda-device 0
```
To run Transduction+GloVe:
```
$ PYTHONHASHSEED=23 python run.py predict predict saved_models/GloVe/model.tar.gz data/grailqa_v1.0_test_public.json --include-package constrained_seq2seq --include-package constrained_seq2seq_reader --predictor seq2seq --use-dataset-reader -c model_configs/test/glove_vp.jsonnet --output-file glove_vp.txt --cuda-device 0
```

### Entity Linking
We also provide instructions on reproduce our entity linking results to benefit future research. Similar to most existing KBQA methods, entity linking is a separate module from our main model. If you just want to run our main models, you do not need to re-run our entity linking module because our models directly retrieve the produce entity linking results under `entity_linking/`.
(To be continued...)


## Train New Models
You can also use our code to train new models.
### Training Configuration
Following [AllenNLP](https://github.com/allenai/allennlp), our `train` command also takes a configuration file as input to specify all model hyperparameters and training related parameters such as learning rate, batch size, cuda device, etc. Most parameters in the training configuration files (i.e., files under `model_configs/train/`) are quite intutive based on their names, so we will only explain those parameters that might be confusing here.
```
ranking: Ranking model or generation mode. True for Ranking, and false for Transduction.
offline: Whether to use cached files under cache/.
num_constants_per_group: Number of schema items in each chunk for BERT encoding.
gq1: True for GraphQuestions, and false for GrailQA.
use_sparql: Whether to use SPARQL as the target query. Set to be false, because in this paper we are using S-expressions.
use_constrained_vocab: Whether to do vocabulary pruning or not.
constrained_vocab: If we do vocabulary pruning, how to do it? Options include 1_step, 2_step and mix2.
perfect_entity_linking: Whether to assume gold entities are given.
```
### Training Command
To train the BERT-based model, run:
```
$ PYTHONHASHSEED=23 python run.py train model_configs/train/train_bert.jsonnet --include-package bert_constrained_seq2seq --include-package bert_seq2seq_reader --include-package utils.bert_interface -s [your_path_specified_for_training]
```
To train the GloVe-based model, run:
```
$ PYTHONHASHSEED=23 python run.py train model_configs/train/train_glove.jsonnet --include-package constrained_seq2seq --include-package constrained_seq2seq_reader -s [your_path_specified_for_training]
```

### Online Running Time
We also show the running time of inference in online mode, in which offline caches are disabled. The aim of this setting is to mimic the real scenario in production. To report the average running time, we random sample 1,000 test questions for each model and run every model on a single 2080-ti GPU card with batch size 1.
|                        | Transduction | Transduction-BERT | Transduction-VP | Transduction-BERT-VP | Ranking | Ranking-BERT |
|------------------------|--------------|-------------------|-----------------|----------------------|---------|--------------|
| Running time (seconds) | 60.899       | 50.176            | 4.787           | 1.932                | 115.459 | 80.892       |


We can see that the running time is quite long when either ranking mode or vocabulary pruning is activated. This is because running SPARQL queries to query the 2-hop information (i.e., either candidate logical forms for ranking or 2-hop schema items for vocabulary pruning) is very time-consuming. This is also a general issue for the enumeration+ranking framework in KBQA, which is used by many existing methods. This issue has to some extend been underaddressed. A common practice is to use offline cache to store the exectuions of all related SPARQL queries, which assumes the test questions are known in advance (this assumption is true for existing KBQA benchmarks, but this does not necessarily mean it is realistic in production).

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
