# This file contains seq2seq models with an extra vocab as input, the
# decoding process are constrained to the input vocab. The only difference
# between models in this file and the original allennlp seq2seq models is
# there is a vocab mask during decoding for each instance.

# Using surface name for entity embedding doesn't really improve the performance. It does help to distinguish different
# entities in the same query, but it makes it difficult for the model to be aware of whether it's an entity or not.


from typing import Dict, List, Tuple
from utils.logic_form_util import same_logical_form, lisp_to_sparql

import numpy
import re
import json
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.models.encoder_decoders import copynet_seq2seq
from allennlp.nn import util
from allennlp.training.metrics import Average
from allennlp.nn.beam_search import BeamSearch

from utils.sparql_executer import execute_query


@Model.register("cons_simple_seq2seq")
class Constrained_SimpleSeq2Seq(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    # Parameters

    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_namespace : ``str``, optional (default = "tokens")
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function : ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    """

    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            target_word_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            max_decoding_steps: int,
            attention: Attention = None,
            attention_function: SimilarityFunction = None,
            beam_size: int = None,
            target_namespace: str = "tokens",
            target_embedding_dim: int = None,
            scheduled_sampling_ratio: float = 0.0,
            # use_constrained_vocab: bool = True,
            ranking_mode: bool = False,  # two different modes, i.e., generation and ranking
            eval=False,
            experiment_sha="default_test"
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._num_schema_items = len(self.vocab._index_to_token[target_namespace])

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # self._entity_names = json.load(open('cache/entity_names.json'))
        self._entity_names = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 10
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        self._target_word_embedder = target_word_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        num_words = self.vocab.get_vocab_size('tgt_words')

        self._exact_match = Average()

        self._exact_match_k = Average()

        self._F1 = Average()

        self._MRR_k = Average()

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError(
                    "You can only specify an attention module or an "
                    "attention function, but not both."
                )
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
            self._output_projection_layer = Linear(self._decoder_output_dim + self._encoder_output_dim,
                                                   target_embedding_dim)
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim
            self._output_projection_layer = Linear(self._decoder_output_dim, target_embedding_dim)

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_vocab = None
        self._output_vocab_mask = None

        self._output_embedding = None
        self._values = None
        self._device = None
        self._batch_size = None
        # self._use_constrained_vocab = use_constrained_vocab
        self._ranking_mode = ranking_mode

        self._eval = eval
        self._experiment_sha = experiment_sha

        self._for_add_graph = False

    def take_step(
            self,
            last_predictions: torch.Tensor,
            state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        # Parameters

        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        # Returns

        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)
        group_size = output_projections.shape[0]
        vocab_mask = state["vocab_mask"]
        # The following code is not necessary. BeamSearch will take care of the shape itself. The only thing
        # is to guarantee every element in state has original shape (batch_size, *)
        # batch_size = vocab_mask.shape[0]
        # num_classes = vocab_mask.shape[1]
        # #  I made a serious mistake here, but if the batch size is 1 then it has no impact.
        # # vocab_mask = vocab_mask.repeat(group_size // batch_size, 1)
        # vocab_mask = vocab_mask.repeat(1, group_size // batch_size)
        # vocab_mask = vocab_mask.reshape(-1, num_classes)
        output_projections.masked_fill_(vocab_mask == 0, -1e32)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(
            self,  # type: ignore
            source_tokens: Dict[str, torch.LongTensor],
            target_tokens: Dict[str, torch.LongTensor] = None,
            target_words=None,
            constrained_vocab=None,
            answer=None,
            values=None,
            value_indices=None,
            candidates=None,
            candidates_value_indices=None,
            entity_name=None,
            ids=None,
            epoch_num=None  # use epoch_num[0] to get the integer epoch number
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        Dict[str, torch.Tensor]
        """
        self._entity_names = entity_name[0]
        for en in entity_name:
            self._entity_names.update(en)
        self._output_vocab, self._output_vocab_mask \
            = self._make_embedder_input(
            self._get_target_tokens(torch.arange(0, self.vocab.get_vocab_size(self._target_namespace))))

        state = self._encode(source_tokens)

        device = source_tokens["tokens"].device
        self._device = device
        # self._batch_size = target_tokens["tokens"].shape[0]
        self._batch_size = len(ids)
        self._values = values
        # Scatter support broadcasting for source tensor
        if not isinstance(constrained_vocab, list):  # using constrained_vocab, tensor instead of list
            # (batch_size, num_classes)
            vocab_mask = torch.zeros(self._batch_size, self.vocab.get_vocab_size(self._target_namespace)) \
                .to(device)
            vocab_mask.scatter_(1, constrained_vocab["tokens"], 1)
        else:  # use_constrained_vocab is False in dataset reader
            # (batch_size, num_classes)
            vocab_mask = torch.ones(self._batch_size, self.vocab.get_vocab_size(self._target_namespace)) \
                .to(device)
            vocab_mask[:, 0] = 0
            vocab_mask[:, 1] = 0  # set @@UNKNOWN@@ to be masked

        # set padding to be 0. padding always corresponds to index 0
        vocab_mask.scatter_(1, torch.zeros(self._batch_size)[:, None].long().to(device), 0)

        value_mask = self._get_value_mask(values)

        state["vocab_mask"] = torch.cat((vocab_mask, value_mask), dim=1)
        # state["vocab_mask"] = vocab_mask

        # (batch_size, 1)
        state["batch_id"] = torch.arange(0, self._batch_size).unsqueeze(-1).to(self._device)

        # (num_classes, decoder_output_dim)
        output_embedding = self._compute_target_embedding(self._output_vocab, self._output_vocab_mask)
        # (decoder_output_dim, num_classes)
        self._output_embedding = output_embedding.transpose(0, 1)

        # (batch_size, output_dim, num_values)
        state["value_embedding"] = self._compute_value_embedding(values, value_mask.shape[1])

        if target_tokens and not self._eval:
            target_tokens["tokens"] = target_tokens["tokens"].float()
            target_tokens["tokens"][value_indices != -1] = self._num_schema_items + value_indices[value_indices != -1]
            state = self._init_decoder_state(state)

            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens, candidates)

            if self.training:
                for i, prediction in enumerate(output_dict['predictions']):
                    self._exact_match(
                        self._compute_exact_match(prediction, target_tokens["tokens"][i], source_tokens["tokens"][i],
                                                  ids[i], values[i]))

            elif self._for_add_graph:
                return output_dict['predictions']
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)

            if not self._ranking_mode:
                #  AllenNLP's beam search returns no more than beam_size of finished states
                predictions = self._forward_beam_search(state)

                output_dict.update(predictions)
                output_dict['values'] = values
                output_dict['ids'] = ids
                # self._output_predictions(predictions['predictions'])

                if not self._eval:  # Do the following operations during validation, but not inference
                    for i, prediction in enumerate(predictions['predictions']):
                        em = self._compute_exact_match(prediction[0],
                                                       target_tokens["tokens"][i],
                                                       source_tokens["tokens"][i],
                                                       ids[i],
                                                       values[i])
                        self._exact_match(em)
                    # if self._eval:
                    #     if em == 1:
                    #         self._F1(1)
                    #     else:
                    #         self._F1(self._compute_F1(prediction[0], values[i], answer[i]))
                    # source_tokens['bert'][i]))
                    # print(self.decode(output_dict)["predicted_tokens"])

            else:
                candidates["tokens"] = candidates["tokens"].float()
                candidates["tokens"][candidates_value_indices != -1] = self._num_schema_items + \
                                                                       candidates_value_indices[
                                                                           candidates_value_indices != -1]
                candidates["tokens"] = candidates["tokens"].long()

                predictions = self._forward_loop_rank(state, candidates)
                output_dict.update(predictions)
                output_dict['values'] = values
                output_dict['ids'] = ids

                if not self._eval:  # Do the following operations during validation, but not inference
                    for i, prediction in enumerate(output_dict['predictions']):
                        em = self._compute_exact_match(prediction,
                                                       target_tokens["tokens"][i],
                                                       source_tokens["tokens"][i],
                                                       ids[i],
                                                       values[i])
                        self._exact_match(em)
                        # if self._eval:
                        #     if em == 1:
                        #         self._F1(1)
                        #     else:
                        #         self._F1(self._compute_F1(prediction, values[i], answer[i]))
                        # source_tokens['bert'][i]))

                    for i, prediction_k in enumerate(output_dict['predictions_k']):
                        em_k, mrr_k = self._compute_exact_match_k(prediction_k,
                                                                  target_tokens["tokens"][i],
                                                                  values[i])
                        self._exact_match_k(em_k)
                        self._MRR_k(mrr_k)

        return output_dict

    @DeprecationWarning
    def _output_predictions(self, predictions):
        """
        Out put the best predicted logical form for each batch instance
        :param predictions: (batch_size, beam_size, num_decoding_steps)
        :return:
        """
        for prediction in predictions:
            logical_form = []
            for token_id in prediction[0]:
                logical_form.append(self.vocab.get_token_from_index(token_id.item(), self._target_namespace))
            rtn = logical_form[0]
            for i in range(1, len(logical_form)):
                if logical_form[i] == '@end@':
                    break
                if logical_form[i - 1] == '(' or logical_form[i] == ')':
                    rtn += logical_form[i]
                else:
                    rtn += ' '
                    rtn += logical_form[i]
            print(rtn)

    def _get_utterance(self, token_ids) -> str:
        question = []
        for token_id in token_ids:
            token = self.vocab.get_token_from_index(token_id.item(), "source_tokens")
            # token = self.vocab.get_token_from_index(token_id.item(), "bert")
            if token == '@end@':
                break
            question.append(token)

        return ' '.join(question[1:])

    def _get_logical_form(self, token_ids, values) -> str:
        logical_form = []
        for token_id in token_ids:
            if token_id.item() < self._num_schema_items:
                logical_form.append(self.vocab.get_token_from_index(token_id.item(), self._target_namespace))
            else:
                logical_form.append(values[int(token_id.item()) - self._num_schema_items])
        rtn = logical_form[0]
        for i in range(1, len(logical_form) - 1):  # the last token is an eos
            if logical_form[i] == '@end@':
                break
            if logical_form[i - 1] == '(' or logical_form[i] == ')':
                rtn += logical_form[i]
            else:
                rtn += ' '
                rtn += logical_form[i]

        return rtn

    def _compute_exact_match(self,
                             predicted: torch.Tensor,
                             target: torch.Tensor,
                             source: torch.Tensor,
                             qid,
                             values: List) -> int:
        predicted_logical_form = self._get_logical_form(predicted, values)
        target_logical_form = self._get_logical_form(target[1:], values)  # omit the start symbol

        # experiment_sha = "gq_dm_test_unconstrained_glove"
        # experiment_sha = "gq_dm_test_unconstrained_el_glove"
        # experiment_sha = "gq_dm_test_ranking_glove"
        # experiment_sha = "gq_dm_test_ranking_el_glove"
        # experiment_sha = "gq_dm_test_glove"
        # experiment_sha = "gq_dm_test_el_glove"

        # experiment_sha = "gq_seen_test_unconstrained_glove"
        # experiment_sha = "gq_seen_test_unconstrained_el_glove"
        # experiment_sha = "gq_seen_test_ranking_glove"
        # experiment_sha = "gq_seen_test_ranking_el_glove"
        # experiment_sha = "gq_seen_test_glove"
        # experiment_sha = "gq_seen_test_el_glove"

        # experiment_sha = 'cwq_s2s'

        # experiment_sha = "gq2_glove_vp_el_1012"
        experiment_sha = self._experiment_sha

        # experiment_sha = 'webqsp_ft'
        if same_logical_form(predicted_logical_form, target_logical_form):
            # if self._eval:
            #     print(qid, self._get_utterance(source),
            #           file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
            #     print("p: ", predicted_logical_form,
            #           file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
            #     print("t: ", target_logical_form,
            #           file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
            return 1
        else:
            # if self._eval:
            #     print(qid, self._get_utterance(source),
            #           file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
            #     print("p: ", predicted_logical_form,
            #           file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
            #     print("t: ", target_logical_form,
            #           file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
            #     denotation = []
            #     try:
            #         sparql_query = lisp_to_sparql(predicted_logical_form)
            #         denotation.extend(execute_query(sparql_query))
            #     except Exception:
            #         pass
            #     print("a: ", '\t'.join(denotation),
            #           file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
            return 0

    def _compute_F1(self, predicted: torch.Tensor,
                    values,
                    answer: List[str]):
        predicted_logical_form = self._get_logical_form(predicted, values)
        try:
            sparql_query = lisp_to_sparql(predicted_logical_form)
            denotation = set(execute_query(sparql_query))
            correct = denotation.intersection(set(answer))
            precision = len(correct) / len(denotation)
            recall = len(correct) / len(answer)

            return (2 * precision * recall) / (precision + recall)
        except Exception:
            return 0

    def _compute_exact_match_k(self,
                               predicted_k: torch.Tensor,
                               target: torch.Tensor,
                               values: List) -> int:
        target_logical_form = self._get_logical_form(target[1:], values)  # omit the start symbol
        for i, predicted in enumerate(predicted_k):
            predicted_logical_form = self._get_logical_form(predicted, values)
            if same_logical_form(predicted_logical_form, target_logical_form):
                return 1, 1. / (i + 1)

        return 0, 0

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        values = output_dict["values"]
        ids = output_dict['ids']

        all_predicted_lfs = []
        all_predicted_answers = []
        for indices, value, qid in zip(predicted_indices, values, ids):
            if not self._ranking_mode:
                indices = indices[0]
            predicted_lf = self._get_logical_form(indices, value)

            denotation = []
            try:
                sparql_query = lisp_to_sparql(predicted_lf)
                denotation.extend(execute_query(sparql_query))
            except Exception:
                pass
            all_predicted_answers.append(denotation)
            all_predicted_lfs.append(predicted_lf)

        rtn = {}
        rtn['qid'] = ids
        rtn['logical_form'] = all_predicted_lfs
        rtn['answer'] = all_predicted_answers
        return rtn

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)  # mask to be handled by encoder
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            state["encoder_outputs"], state["source_mask"], self._encoder.is_bidirectional()
        )
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        return state

    #  ranking mode does not use constrained vocabulary
    def _forward_loop_rank(self,
                           state: Dict[str, torch.Tensor],
                           candidates: Dict[str, torch.LongTensor] = None,
                           scoring_fn: str = 'sum',
                           num_for_training: int = 20):  # scoring function can be either sum or avg
        # candidates: shape (batch_size, num_candidates, num_of_tokens). There are paddings
        # along dimension 1 and dimension 2
        if not self.training:
            if candidates["tokens"].shape[1] > 300:
                num_splits = candidates["tokens"].shape[1] // 300 + 1
                log_probs_sum_splits = []
                log_probs_avg_splits = []
                for i in range(num_splits - 1):
                    log_probs_sum_i, log_probs_avg_i = self._computing_one_candidates_shard(
                        candidates["tokens"][:, i * 300: (i + 1) * 300], state)
                    log_probs_sum_splits.append(log_probs_sum_i)
                    log_probs_avg_splits.append(log_probs_avg_i)
                if (i + 1) * 300 < candidates["tokens"].shape[1]:
                    log_probs_sum_i, log_probs_avg_i = self._computing_one_candidates_shard(
                        candidates["tokens"][:, (i + 1) * 300:], state)
                    log_probs_sum_splits.append(log_probs_sum_i)
                    log_probs_avg_splits.append(log_probs_avg_i)

                log_probs_sum = torch.cat(log_probs_sum_splits, dim=1)
                log_probs_avg = torch.cat(log_probs_avg_splits, dim=1)

            else:
                log_probs_sum, log_probs_avg = self._computing_one_candidates_shard(candidates["tokens"], state)
            # num_candidates = candidates["tokens"].shape[1]
        else:
            candidate_shard = candidates["tokens"][:, num_for_training]
            log_probs_sum, log_probs_avg, logits = self._computing_one_candidates_shard(candidate_shard, state)
            # num_candidates = min(candidates["tokens"].shape[1], num_for_training)

        if not self.training:
            targets = candidates["tokens"]
        else:
            targets = candidates["tokens"][:, :num_for_training]
        targets = targets[:, :, 1:].contiguous()
        batch_size, num_of_candidates, target_sequence_length = targets.size()
        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        assert scoring_fn in ['sum', 'avg']
        if scoring_fn == 'sum':
            # (batch_size, )
            best_lfs = torch.argmax(log_probs_sum, dim=1)
        else:
            best_lfs = torch.argmax(log_probs_avg, dim=1)
        # (batch_size, num_decoding_steps)  TODO: check whether there is a better way to do this kind of indexing
        predictions = targets[torch.arange(batch_size).to(self._device), best_lfs]

        k = min(10, log_probs_sum.shape[1])
        # best_k_lfs: (batch_size, k)
        _, best_k_lfs = log_probs_sum.topk(k, dim=-1)

        index_0 = torch.arange(batch_size).unsqueeze(-1).repeat(1, k).to(self._device)
        # (batch_size, k, num_decoding_steps)
        predictions_k = targets[index_0, best_k_lfs]

        return {"predictions": predictions, "predictions_k": predictions_k}

    def _computing_one_candidates_shard(self, candidates_shard: torch.Tensor, state) -> (torch.Tensor, torch.Tensor):
        num_candidates = candidates_shard.shape[1]

        new_state = {}

        # shape: (batch_size * num_candidates, decoder_output_dim)
        new_state["decoder_hidden"] = state["decoder_hidden"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(-1, self._decoder_output_dim)

        new_state["decoder_context"] = state["decoder_context"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(-1, self._decoder_output_dim)

        # (batch_size * num_candidates, max_input_sequence_length, encoder_output_dim)
        new_state["encoder_outputs"] = state["encoder_outputs"].unsqueeze(1) \
            .repeat(1, num_candidates, 1, 1).reshape(self._batch_size * num_candidates,
                                                     -1,
                                                     self._encoder_output_dim)
        # (batch_size * num_candidates, max_input_sequence_length)
        new_state["source_mask"] = state["source_mask"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(self._batch_size * num_candidates, -1)

        num_values = state["value_embedding"].shape[-1]
        # (batch_size * num_candidates, embedding_dim, num_values)
        new_state["value_embedding"] = state["value_embedding"].unsqueeze(1) \
            .repeat(1, num_candidates, 1, 1).reshape(self._batch_size * num_candidates,
                                                     -1,
                                                     num_values)
        # (batch_size * num_candidates, 1)
        new_state["batch_id"] = state["batch_id"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(self._batch_size * num_candidates, -1)

        vocab_mask = state["vocab_mask"]
        # shape: (batch_size, 1, num_classes)
        vocab_mask = vocab_mask.unsqueeze(1)

        # shape: (batch_size, num_of_candidates, num_decoding_steps)
        targets = candidates_shard

        batch_size, num_of_candidates, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        step_logits: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size, num_of_candidates)
            input_choices = targets[:, :, timestep]
            # (batch_size * num_of_candidates)
            input_choices = input_choices.reshape(batch_size * num_of_candidates, )
            # shape: (batch_size * num_of_candidates, num_classes)
            output_projections, new_state = self._prepare_output_projections(input_choices, new_state)

            output_projections = output_projections.reshape(batch_size, num_of_candidates, -1)

            # apply the vocab mask
            output_projections.masked_fill_(vocab_mask == 0, -1e32)
            # shape: (batch_size * num_of_candidates, num_classes)
            output_projections = output_projections.reshape(batch_size * num_of_candidates, -1)

            # list of tensors, shape: (batch_size * num_of_candidates, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

        # shape: (batch_size * num_of_candidates, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        # shape: (batch_size, num_of_candidates, num_decoding_steps)
        # 0 stands for padding, 1 stands for unmasked
        target_mask = util.get_text_field_mask({"tokens": candidates_shard}, num_wrapping_dims=1)[:, :num_candidates, :]

        # Note we don't need to predict a probability for <sos>, so there is an offset of size 1
        targets = targets[:, :, 1:].contiguous()
        target_mask = target_mask[:, :, 1:].contiguous()

        # shape: (batch_size, num_of_candidates)
        target_len = target_mask.sum(dim=-1)
        if target_len.shape[1] > 250:
            # (batch_size, num_of_candidates, num_decoding_steps, num_classes)
            logits = logits.reshape(batch_size, num_of_candidates, num_decoding_steps, -1)
            num_splits = target_len.shape[1] // 250 + 1

            log_probs_gather_split = []
            for i in range(num_splits - 1):
                logits_i = logits[:, i * 250: (i + 1) * 250]
                # (batch_size, 300, num_decoding_steps, num_classes)
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                # (batch_size, 300, num_decoding_steps, 1)
                # print('targets shape: ', targets.shape)
                log_probs_gather_i = log_probs_i.gather(dim=-1,
                                                        index=targets[:, i * 250: (i + 1) * 250].unsqueeze(
                                                            -1))
                # doesn't really help
                # del logits_i
                # torch.cuda.empty_cache()
                log_probs_gather_i = log_probs_gather_i.squeeze(-1)
                log_probs_gather_split.append(log_probs_gather_i)
            logits_i = logits[:, (i + 1) * 250:]
            # (batch_size, left_candidates, num_decoding_steps, num_classes)
            log_probs_i = F.log_softmax(logits_i, dim=-1)
            # (batch_size, left_candidates, num_decoding_steps, 1)
            log_probs_gather_i = log_probs_i.gather(dim=-1, index=targets[:, (i + 1) * 250:].unsqueeze(-1))
            log_probs_gather_i = log_probs_gather_i.squeeze(-1)
            log_probs_gather_split.append(log_probs_gather_i)

            log_probs_gather = torch.cat(log_probs_gather_split, dim=1)

        else:
            # shape: (batch_size * num_of_candidates, num_decoding_steps, num_classes)
            log_probs = F.log_softmax(logits, dim=-1)
            # (batch_size, num_of_candidates, num_decoding_steps, num_classes)
            log_probs = log_probs.reshape(batch_size, num_of_candidates, num_decoding_steps, -1)
            # (batch_size, num_of_candidates, num_decoding_steps, 1)
            log_probs_gather = log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
            log_probs_gather = log_probs_gather.squeeze(-1)

        log_probs_gather = log_probs_gather * target_mask
        # (batch_size, num_of_candidates)
        log_probs_sum = log_probs_gather.sum(dim=-1)
        log_probs_sum.masked_fill_(log_probs_sum == 0, -1e32)

        log_probs_avg = log_probs_sum / target_len

        if not self.training:
            return log_probs_sum, log_probs_avg
        else:
            return log_probs_sum, log_probs_avg, logits

    def log_graph(self) -> None:
        self._for_add_graph = True

    def _forward_loop(
            self,
            state: Dict[str, torch.Tensor],
            target_tokens: Dict[str, torch.LongTensor] = None,
            candidates: Dict[str, torch.LongTensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, num_classes)
        vocab_mask = state["vocab_mask"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"]

        _, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.  (In fact, I beleive this comment from ai2 is incorrect,
        # the real reason is we don't have to process the <sos> token)
        num_decoding_steps = target_sequence_length - 1

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((self._batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # apply the vocab mask
            output_projections.masked_fill_(vocab_mask == 0, -1e32)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)

            output_dict["loss"] = loss  # Here is the only place that loss being calculated

        return output_dict

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # can be used to compute exact match
        # shape (log_probabilities): (batch_size, beam_size), the probability of generating
        # the associated sequence
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]
                                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # (group_size, embedding_dim, num_values)
        value_embedding = state["value_embedding"]

        # shape: (group_size, target_embedding_dim)
        # embedded_input = self._target_embedder(last_predictions)
        # We need an extra batch_id argument here because now we need the batch_id to get the proper surface name for
        # entities and literals. While previously, everything is taken from a global static vocab, so there is no need
        # for batch_id.
        converted_input, mask = self._make_embedder_input(self._get_target_tokens(last_predictions, state["batch_id"]))
        embedded_input = self._compute_target_embedding(converted_input, mask)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(
                decoder_hidden, encoder_outputs, source_mask
            )

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # TODO: use attention also for logits projection not just input feeding
        if self._attention:
            decoder_output = torch.cat((attended_input, decoder_hidden), -1)
        else:
            decoder_output = decoder_hidden

        output_embedding = self._output_embedding

        projected_hidden = self._output_projection_layer(
            decoder_output)  # one more linear layer before computing logits

        # (batch_size, num_schema_items)
        output_projections = torch.mm(projected_hidden, output_embedding)

        projected_hidden = projected_hidden.unsqueeze(1)
        output_projections_values = torch.bmm(projected_hidden, value_embedding)
        # (batch_size, num_values)
        output_projections_values = output_projections_values.squeeze(1)

        output_projections = torch.cat((output_projections, output_projections_values), dim=-1)

        return output_projections, state

    def _compute_target_embedding(self,
                                  x: torch.Tensor,
                                  mask: torch.Tensor,
                                  pooling: str = None) -> torch.Tensor:
        pooling = pooling or 'mean'
        x = x.to(self._device)
        mask = mask.to(self._device)

        # (batch_size, num_of_words, embedding_dim)
        embeddings = self._target_word_embedder({"tokens": x})  # Note here "tokens" is specified to match the embedder
        # (batch_size, num_of_words, embedding_dim)
        mask = (mask.unsqueeze(-1)).expand(-1, -1, embeddings.shape[-1])
        # (batch_size, num_of_words, embedding_dim)
        embeddings = embeddings * mask

        if pooling == 'sum':
            # (batch_size, embedding_dim)
            embeddings = embeddings.sum(1)
        elif pooling == 'mean':
            mask = mask[:, :, 0].sum(1).unsqueeze(1)
            embeddings = embeddings.sum(1)
            embeddings = embeddings / mask

        # (batch_size, embedding_dim)
        return embeddings

    def _compute_value_embedding(self, values: List, length_to_pad: int):
        # Return a tensor of shape (batch_size, embedding_dim, length_to_pad)
        embeddings = []
        for i in range(length_to_pad):
            # Compute (batch_size, embedding_dim, 1) at each step
            tokens_i = []
            for value in values:
                if i < len(value):
                    if value[i] in self._entity_names:
                        tokens_i.append(value[i])
                    else:  # literal
                        tokens_i.append(value[i].lower())
                else:
                    tokens_i.append("pad")  # whatever. doesn't matter, it will be masked anyway
            converted_input, mask = self._make_embedder_input(tokens_i)
            # (batch_size, embedding_dim, 1)
            embeddings_i = self._compute_target_embedding(converted_input, mask)
            embeddings.append(embeddings_i)
        return torch.stack(embeddings, dim=2)

    def _prepare_attended_input(
            self,
            decoder_hidden_state: torch.LongTensor = None,
            encoder_outputs: torch.LongTensor = None,
            encoder_outputs_mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
            logits: torch.FloatTensor,
            targets: torch.LongTensor,
            target_mask: torch.LongTensor,
            average: str = "batch"
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # # shape: (batch_size, num_decoding_steps)
        # relevant_targets = targets[:, 1:].contiguous()
        #
        # # shape: (batch_size, num_decoding_steps)
        # relevant_mask = target_mask[:, 1:].contiguous()
        #
        # return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

        # Instead of using the interface provided by util.sequence_cross_entropy_with_logits, we define the computation
        # here to make it more flexible.

        # shape: (batch_size, num_decoding_steps)
        targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        weights = target_mask[:, 1:].contiguous()

        # make sure weights are float
        weights = weights.float()
        # sum all dim except batch
        non_batch_dims = tuple(range(1, len(weights.shape)))
        # shape : (batch_size,)
        weights_batch_sum = weights.sum(dim=non_batch_dims)
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights

        if average == "batch":
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences
        elif average == "token":
            return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
        else:
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            return per_batch_loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # reset is set to be True by default in trainer
        all_metrics: Dict[str, float] = {}
        all_metrics['example_count'] = self._exact_match._count
        all_metrics['exact_match_count'] = self._exact_match._total_value
        all_metrics['exact_match'] = self._exact_match.get_metric(reset)
        all_metrics['exact_match_k'] = self._exact_match_k.get_metric(reset)
        all_metrics['F1'] = self._F1.get_metric(reset)
        all_metrics['MRR_k'] = self._MRR_k.get_metric(reset)
        return all_metrics

    def _get_value_mask(self, values: List):
        max_len = 0
        for values_i in values:
            if len(values_i) > max_len:
                max_len = len(values_i)
        value_mask = torch.zeros(self._batch_size, max_len).to(self._device)

        for i, values_i in enumerate(values):
            value_mask[i][: len(values_i)] = 1

        return value_mask

    def _get_target_tokens(self, x, batch_id=None):
        tokens = []
        for i, id in enumerate(x):
            if id.item() < self._num_schema_items:
                token = self.vocab._index_to_token[self._target_namespace][id.item()]
            else:
                token = self._values[batch_id[i]][int(id.item()) - self._num_schema_items]
            tokens.append(token)
        return tokens

    def _make_embedder_input(self, x):
        """
        Convert a list of logical constant indexes into an input to the word-level embedder
        :param x: (group_size, )
        :return: converted_input: (group_size, num_words), mask: (group_size, num_words)
        Here mask use 1 to indicate being used and 0 for masked, which is different from
        vocab_mask
        """
        group_size = len(x)
        tokens_list = []
        max_len = 0
        for token in x:
            if token in self._entity_names:
                token = self._entity_names[token]
                # token = 'entity'
                token_words = re.split('[._ ]', token)[:5]
            elif token.__contains__('^^'):
                token_words = token.split('^^')[0].replace('"', '').split('-')
            else:
                token_words = re.split('[._ ]', token)
            tokens_list.append(token_words)
            max_len = max(max_len, len(token_words))

        mask = torch.zeros(group_size, max_len)
        converted_input = []
        for i, token_words in enumerate(tokens_list):
            word_ids = []
            for word in token_words:
                if word in self.vocab._token_to_index['tgt_words']:
                    word_ids.append(self.vocab._token_to_index['tgt_words'][word])
                else:
                    word_ids.append(self.vocab._token_to_index['tgt_words']['@@UNKNOWN@@'])
            mask[i][:len(word_ids)] = 1
            for _ in range(max_len - len(word_ids)):
                word_ids.append(0)
            converted_input.append(word_ids)

        converted_input = torch.tensor(converted_input)

        return converted_input, mask
