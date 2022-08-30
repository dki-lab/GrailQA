local ranking = false;
local num_constants_per_group = 1;
local offline = true;
local gq1 = false;
local cuda = 0;
local use_sparql = false;
{
  "dataset_reader":{
    "type":"bert_seq2seq",
    "gq1": gq1,
    "offline": offline,
    "constrained_vocab":"mix2",
    "use_sparql": use_sparql,
    "training": true,
    "source_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "bert-base-uncased",
      "do_lowercase": true
    },
    "source_token_indexers":{
      "bert": {
        "type": "pretrained_transformer",
        "model_name": "bert-base-uncased",
        "do_lowercase": true,
        "namespace": "bert"
      }
    },
    "target_token_indexers":{"tokens": {"namespace": "target_tokens"}},
    "num_constants_per_group": num_constants_per_group
  },
  "validation_dataset_reader":{
    "type":"bert_seq2seq",
    "gq1": gq1,
    "offline": offline,
    "constrained_vocab":"2_step",
    "use_sparql": use_sparql,
    "perfect_entity_linking": true,
    "ranking_mode": ranking,
    "training": false,
    "source_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "bert-base-uncased",
      "do_lowercase": true
    },
    "source_token_indexers":{
      "bert": {
        "type": "pretrained_transformer",
        "model_name": "bert-base-uncased",
        "do_lowercase": true,
        "namespace": "bert"
      }
    },
    "target_token_indexers":{"tokens": {"namespace": "target_tokens"}},
    "num_constants_per_group": num_constants_per_group
  },
  "train_data_path": "data/finetune/one_neighbor/finetune_oneneighbor_questions_trainshort.json",
  "validation_data_path": "data/finetune/one_neighbor/finetune_oneneighbor_questions_valshort.json",
  "model": {
    "type": "bert_cons_simple_seq2seq",
    "use_sparql": use_sparql,
    "source_embedder": {
      "allow_unmatched_keys": true,
        "bert": {
          "type": "my_pretrained_transformer",
          "model_name": "bert-base-uncased"
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 384,  //  bidirectional 384, unidirectional 768
      "bidirectional": true,
      "num_layers": 1
    },
    "target_embedding_dim": 768,
    "max_decoding_steps": 200,
    "target_namespace": "target_tokens",
    "ranking_mode": ranking,
    "attention_function": {"type": "dot_product"},
    "num_constants_per_group": num_constants_per_group
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size" : 1,
    "track_epoch" : true
  },
  "trainer": {
    "num_epochs": 2,
    "validation_metric": "+exact_match",
    "patience": 2,
    "cuda_device": cuda,
    "num_gradient_accumulation_steps": 16,
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "parameter_groups": [
        [["source_embedder"], {"lr": 2e-5}]
      ]
    }
  }
}
