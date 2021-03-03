local ranking = false;
local offline = true;
local cuda = 4;

{
  "dataset_reader": {
    "type": "cons_seq2seq",
    "device": cuda,
    "constrained_vocab": "mix2",
    "offline": offline,
    "training": true,
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    },
    "use_constrained_vocab": true
  },
  "validation_dataset_reader": {
    "type": "cons_seq2seq",
    "device": cuda,
    "constrained_vocab": "2_step",
    "offline": offline,
    "training": false,
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    },
    "ranking_mode": ranking,
    "use_constrained_vocab": true
  },
  "vocabulary": {
     "directory_path": "vocabulary/graphq"
  },
    "train_data_path": "data/grailqa_v1.0_train.json",
  "validation_data_path": "data/grailqa_v1.0_dev.json",
  "model": {
    "type": "cons_simple_seq2seq",
    "source_embedder": {
      "allow_unmatched_keys": true,
      "tokens": {
        "type": "embedding",
        "vocab_namespace": "source_tokens",
        "embedding_dim": 300,
        "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
        "trainable": false
      }

    },
    "target_word_embedder": {
      "tokens": {
        "type": "embedding",
        "vocab_namespace": "tgt_words",
        "embedding_dim": 300,
        "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
        "trainable": false
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      //1024
      "hidden_size": 768,
      //512
      //sometimes it might be a bad idea to set hidden_size to be much smaller than input_size
      "num_layers": 1
    },
    "target_embedding_dim": 300,
    "max_decoding_steps": 100,
    "target_namespace": "target_tokens",
    "attention_function": {
      "type": "dot_product"
    },
    "ranking_mode": ranking
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size": 32,
    "track_epoch": true
  },
  "trainer": {
    "num_epochs": 45,
    "validation_metric": "+exact_match",
    "patience": 5,
    "cuda_device": cuda,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}