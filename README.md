# GrailQA: Strongly <ins>G</ins>ene<ins>ra</ins>l<ins>i</ins>zab<ins>l</ins>e Question Answering
<img width="1175" alt="image" src="https://user-images.githubusercontent.com/15921425/110228546-f2193380-7ecf-11eb-8cbd-c5097a064ee4.png">

GrailQA is a new large-scale, high-quality KBQA dataset with 64,331 questions annotated with both answers and corresponding logical forms in different syntax (i.e., SPARQL, S-expression, etc.). It can be used to test three levels of generalization in KBQA: **i.i.d.**, **compositional**, and **zero-shot**.

For dataset and leaderboard, please refer to the [homepage of GrailQA](https://dki-lab.github.io/GrailQA/). In this repo, we help you to reproduce the results of our baseline models and to train new models using our code.

## Overview
To study the three levels of generalization in KBQA, we implement a line of baseline models of different natures, namely, **Transduction+BERT**, **Transduction+GloVe**, **Ranking+BERT**, and **Ranking+GloVe**. Our implementation is based on [AllenNLP](https://github.com/allenai/allennlp). 

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
### Reproduce Main Results
### Reproduce Entity Linking Results


## Train New Models
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
