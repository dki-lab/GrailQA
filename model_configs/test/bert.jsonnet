local ranking = false;
local num_constants_per_group = 30;
local offline = true;
{
  "dataset_reader":{
    "type":"bert_seq2seq",
    "offline": offline,
    "constrained_vocab":"mix2",
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
    "lazy": true,
    "offline": offline,
    "constrained_vocab":"2_step",
    "perfect_entity_linking": false,
    "ranking_mode": ranking,
    "use_constrained_vocab": false,
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
  "train_data_path": "data/grailqa_v1.0_train.json",
  "validation_data_path": "data/grailqa_v1.0_dev.json",
  "model": {
    "eval": true,
    "experiment_sha": "",
    "type": "bert_cons_simple_seq2seq",
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
    "max_decoding_steps": 100,
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
    "num_epochs": 20,
    "validation_metric": "+exact_match",
    "patience": 3,
    "cuda_device": 1,
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
