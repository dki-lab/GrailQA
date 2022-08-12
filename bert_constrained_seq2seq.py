from typing import Dict, List, Tuple
from utils.logic_form_util import same_logical_form, lisp_to_sparql
from utils.sparql_executer import execute_query

import numpy
import re
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from allennlp.modules.token_embedders import pretrained_transformer_embedder
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import Average
from allennlp.nn.beam_search import BeamSearch
from allennlp.training import trainer
import pickle


@Model.register("bert_cons_simple_seq2seq")
class Bert_Constrained_SimpleSeq2Seq(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            max_decoding_steps: int,
            attention: Attention = None,
            attention_function: SimilarityFunction = None,
            beam_size: int = None,
            target_namespace: str = "tokens",
            target_embedding_dim: int = None,
            ranking_mode: bool = False,
            scheduled_sampling_ratio: float = 0.0,
            num_constants_per_group=45,
            eval=False,
            use_sparql=False,
            experiment_sha="default_test"
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._use_sparql = use_sparql

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        # Because we don't have a global vocabulary, instead, we only have a dynamic constrained
        # vocabulary for each instance, so we force the end and start symbol to be the first two
        # tokens in our constrained vocab during data loading
        if not self._use_sparql:
            self._start_index = 12
            self._end_index = 13
        else:
            self._start_index = 37
            self._end_index = 38

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 10
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._exact_match = Average()

        self._F1 = Average()

        self._exact_match_k = Average()

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
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_embedding = None
        self._device = None
        self._vocab_size = None

        self._ranking_mode = ranking_mode
        self._num_constants_per_group = num_constants_per_group

        self._eval = eval
        self._experiment_sha = experiment_sha

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
        # batch_size = vocab_mask.shape[0]
        # num_classes = vocab_mask.shape[1]
        # # vocab_mask = vocab_mask.repeat(group_size // batch_size, 1)   #  I made a serious mistake here
        # vocab_mask = vocab_mask.repeat(1, group_size // batch_size)
        # vocab_mask = vocab_mask.reshape(-1, num_classes)
        # pay attention whether 0 or 1 denotes mask (by convention it should be 0)
        output_projections.masked_fill_(vocab_mask == 0, -1e32)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(
            self,  # type: ignore
            source_tokens: Dict[str, torch.LongTensor],
            target_tokens: torch.LongTensor = None,
            schema_start: List = None,
            schema_end: List = None,
            constrained_vocab=None,
            answer=None,
            ids=None,
            candidates=None,
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

        # torch.autograd.set_detect_anomaly(True)  # this option makes training slower
        #print('this is the forward function')
        #print(source_tokens)
        #print(target_tokens)
        # self._vocab_size = constrained_vocab['tokens'].shape[1]
        print('##########################')
        print('target tokens')
        print(target_tokens)
        print('$$$$$$$$$$$$$$$$$$$$$$$$')
        batch_size = len(constrained_vocab)
        self._vocab_size = 0
        for c_v in constrained_vocab:
            if len(c_v) > self._vocab_size:
                self._vocab_size = len(c_v)

        # device = target_tokens.device
        device = source_tokens['bert'].device
        self._device = device

        # shape: (batch_size, num_seq, max_input_sequence_length, encoder_input_dim)
        # embedded_input = self._source_embedder(source_tokens)
        source_shape = source_tokens["bert"].shape
        embedded_input = self._get_bert_embeddings(source_tokens)
        source_tokens["bert"] = source_tokens["bert"].reshape(source_shape)  # restore the shape
        # embedded_input = self._source_embedder({"bert": source_tokens["bert"][:, 0, :]})
        # only use the first concatenation for utterance encoding
        state = self._encode(embedded_input[:, 0, :], schema_start)

        # (batch_size, num_classes)
        # vocab_mask = util.get_text_field_mask(constrained_vocab)
        vocab_mask = torch.zeros(batch_size, self._vocab_size)
        for i, c_v in enumerate(constrained_vocab):
            vocab_mask[i][:len(c_v)] = 1
        vocab_mask = vocab_mask.long().to(self._device)
        state["vocab_mask"] = vocab_mask

        # (batch_size, num_classes, decoder_output_dim)
        output_embedding = self._compute_target_embedding(schema_start,
                                                          schema_end,
                                                          embedded_input)
        # (batch_size, decoder_output_dim, num_classes)
        self._output_embedding = output_embedding.transpose(1, 2)

        if target_tokens is not None and not self._eval:
            target_tokens = target_tokens.squeeze(-1)
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)

            if self.training:
                for i, prediction in enumerate(output_dict['predictions']):
                    self._exact_match(self._compute_exact_match(prediction,
                                                                target_tokens[i],
                                                                # constrained_vocab['tokens'][i],
                                                                constrained_vocab[i],
                                                                ids[i],
                                                                source_tokens['bert'][i][0]))
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)

            if not self._ranking_mode:
                #  AllenNLP's beam search returns no more than beam_size of finished states
                predictions = self._forward_beam_search(state)

                output_dict.update(predictions)
                # self._output_predictions(predictions['predictions'])
                output_dict["constrained_vocab"] = constrained_vocab
                output_dict['ids'] = ids

                if not self._eval:
                    for i, prediction in enumerate(predictions['predictions']):
                        em = self._compute_exact_match(prediction[0],
                                                       target_tokens[i],
                                                       # constrained_vocab['tokens'][i],
                                                       constrained_vocab[i],
                                                       ids[i],
                                                       source_tokens['bert'][i][0])
                        self._exact_match(em)
                        # if self._eval:
                        #     if em == 1:
                        #         self._F1(1)
                        #     else:
                        #         self._F1(self._compute_F1(prediction[0], constrained_vocab[i], answer[i]))

                    for i, prediction_k in enumerate(predictions['predictions']):
                        em_k, mrr_k = self._compute_exact_match_k(prediction_k,
                                                                  target_tokens[i],
                                                                  constrained_vocab[i])
                        self._exact_match_k(em_k)
                        self._MRR_k(mrr_k)

            else:
                # candidates: shape (batch_size, num_of_logical_forms, num_of_tokens). There are paddings
                # along dimension 1 and dimension 2
                candidates = candidates.squeeze(-1)

                predictions = self._rank_candidates(state, candidates)
                output_dict.update(predictions)

                output_dict["constrained_vocab"] = constrained_vocab
                output_dict['ids'] = ids

                if not self._eval:
                    for i, prediction in enumerate(predictions['predictions']):
                        em = self._compute_exact_match(prediction,
                                                       target_tokens[i],
                                                       # constrained_vocab['tokens'][i],
                                                       constrained_vocab[i],
                                                       ids[i],
                                                       source_tokens['bert'][i][0])
                        self._exact_match(em)
                        # if self._eval:
                        #     if em == 1:
                        #         self._F1(1)
                        #     else:
                        #         self._F1(self._compute_F1(prediction, constrained_vocab[i], answer[i]))

                        # source_tokens['bert'][i]))

                # for i, prediction_k in enumerate(predictions['predictions_k']):
                #     em_k, mrr_k = self._compute_exact_match_k(prediction_k,
                #                                               target_tokens[i],
                #                                               constrained_vocab['tokens'][i],
                #                                               source_tokens['bert'][i])
                #
                #     self._exact_match_k(em_k)
                #     self._MRR_k(mrr_k)

        return output_dict

    # It will consume too much memory if computing all concatenated input at the same time.
    # This function simply computes the embeddings one by one for each input
    # Also the new version of BERT api only takes input of shape (batch_size, seq_len)
    def _get_bert_embeddings(self, source_tokens: Dict[str, torch.LongTensor]):
        # This doesn't really help to reduce the memory consumption during training, because the main issue is
        # backpropagation.
        # bert_embeddings = []
        # for i in range(0, source_tokens["bert"].shape[1], 10):
        #     if i + 10 <= source_tokens["bert"].shape[1]:
        #         bert_embeddings.append(self._source_embedder({"bert": source_tokens["bert"][:, i:i+10, :]}))
        #     else:
        #         bert_embeddings.append(self._source_embedder({"bert": source_tokens["bert"][:, i:, :]}))
        #
        # # (batch_size, num_seq, max_len, dim)
        # return torch.cat(bert_embeddings, 1)
        try:
            batch_size = source_tokens["bert"].shape[0]
            num_seq = source_tokens["bert"].shape[1]
            max_len = source_tokens["bert"].shape[2]
            source_tokens["bert"] = source_tokens["bert"].reshape(batch_size * num_seq, -1)
            if self.training:
                
                return self._source_embedder(source_tokens).reshape(batch_size, num_seq, max_len, -1)
            else:
                bert_embeddings = []
                for i in range(0, source_tokens["bert"].shape[0], 10 * batch_size):
                    if (i + 10) * batch_size <= source_tokens["bert"].shape[0]:
                        bert_embeddings.append(self._source_embedder(
                            {"bert": source_tokens["bert"][batch_size * i: batch_size * (i + 10), :]}))
                    else:
                        bert_embeddings.append(
                            self._source_embedder({"bert": source_tokens["bert"][batch_size * i:, :]}))

                # (batch_size, num_seq, max_len, dim)
                return torch.cat(bert_embeddings, 0).reshape(batch_size, num_seq, max_len, -1)
        except MemoryError:
            print("oom sample: ", self._get_utterance(source_tokens['bert'][0][0]), source_tokens['bert'].shape)

    def _rank_candidates(self,
                         state: Dict[str, torch.Tensor],
                         candidates: torch.LongTensor = None,
                         scoring_fn: str = 'avg'):  # scoring function can be either sum or avg
        if candidates.shape[1] > 50:
            num_splits = candidates.shape[1] // 50 + 1
            log_probs_sum_splits = []
            log_probs_avg_splits = []
            for i in range(num_splits - 1):
                log_probs_sum_i, log_probs_avg_i = self._computing_one_candidates_shard(
                    candidates[:, i * 50: (i + 1) * 50], state)
                log_probs_sum_splits.append(log_probs_sum_i)
                log_probs_avg_splits.append(log_probs_avg_i)

            if (i + 1) * 50 < candidates.shape[1]:
                log_probs_sum_i, log_probs_avg_i = self._computing_one_candidates_shard(
                    candidates[:, (i + 1) * 50:], state)
                log_probs_sum_splits.append(log_probs_sum_i)
                log_probs_avg_splits.append(log_probs_avg_i)

            log_probs_sum = torch.cat(log_probs_sum_splits, dim=1)
            log_probs_avg = torch.cat(log_probs_avg_splits, dim=1)
            # try:
            #     log_probs_sum, log_probs_avg = self._computing_one_candidates_shard(candidates, state)
            # except RuntimeError:
            #     print("\nOOM shape: ", candidates.shape)

        else:
            log_probs_sum, log_probs_avg = self._computing_one_candidates_shard(candidates, state)

        targets = candidates
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
        batch_size = candidates_shard.shape[0]

        new_state = {}

        # shape: (batch_size * num_candidates, decoder_output_dim)
        new_state["decoder_hidden"] = state["decoder_hidden"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(-1, self._decoder_output_dim)

        new_state["decoder_context"] = state["decoder_context"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(-1, self._decoder_output_dim)

        # (batch_size * num_candidates, max_input_sequence_length, encoder_output_dim)
        new_state["encoder_outputs"] = state["encoder_outputs"].unsqueeze(1) \
            .repeat(1, num_candidates, 1, 1).reshape(batch_size * num_candidates,
                                                     -1,
                                                     self._encoder_output_dim)
        # (batch_size * num_candidates, max_input_sequence_length)
        new_state["source_mask"] = state["source_mask"].unsqueeze(1) \
            .repeat(1, num_candidates, 1).reshape(batch_size * num_candidates, -1)

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
        target_mask = (candidates_shard != -1).long()

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
            targets.masked_fill_(targets == -1, 0)
            for i in range(num_splits - 1):
                logits_i = logits[:, i * 250: (i + 1) * 250]
                # (batch_size, 250, num_decoding_steps, num_classes)
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                # (batch_size, 250, num_decoding_steps, 1)
                # print('targets shape: ', targets.shape)
                log_probs_gather_i = log_probs_i.gather(dim=-1,
                                                        index=targets[:, i * 250: (i + 1) * 250].unsqueeze(
                                                            -1))
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
            # -1 is an illegal index for gather, replace. fine to replace it with anything as we have mask to ignore it
            # For non-bert model, the padding is 0, so we don't have the same issue.
            targets.masked_fill_(targets == -1, 0)
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
            # token = self.vocab.get_token_from_index(token_id.item(), "source_tokens")
            token = self.vocab.get_token_from_index(token_id.item(), "bert")
            if token == '[SEP]':
                break
            question.append(token)

        return ' '.join(question[1:])

    def _get_logical_form(self,
                          token_ids,
                          constrained_vocab) -> str:
        logical_form = []
        for token_id in token_ids:
            # logical_form.append(
            #     self.vocab.get_token_from_index(constrained_vocab[token_id].item(), self._target_namespace))
            logical_form.append(constrained_vocab[token_id])
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

    def _compute_exact_match_k(self,
                               predicted_k,
                               target,
                               constrained_vocab):
        try:
            target_logical_form = self._get_logical_form(target[1:], constrained_vocab)
        except Exception:  # there might be an exception here when tokens in target is not covered by constrained vocab
            return 0, 0
        for i, predicted in enumerate(predicted_k):
            predicted_logical_form = self._get_logical_form(predicted, constrained_vocab)
            if same_logical_form(target_logical_form, predicted_logical_form):
                return 1, 1. / (i + 1)

        return 0, 0

    def _compute_exact_match(self,
                             predicted: torch.Tensor,
                             target: torch.Tensor,
                             # constrained_vocab: torch.Tensor,
                             constrained_vocab: List[str],
                             qid,
                             source: torch.Tensor = None) -> int:
        predicted_logical_form = self._get_logical_form(predicted, constrained_vocab)
        try:
            target_logical_form = self._get_logical_form(target[1:], constrained_vocab)  # omit the start symbol
        except Exception:
            # return 0
            target_logical_form = "@@UNKNOWN@@"

        # print(predicted_logical_form)
        # print(target_logical_form)


        # experiment_sha = "gq_dm_test_unconstrained"
        # experiment_sha = "gq_dm_test_unconstrained_el"
        # experiment_sha = "gq_dm_test_ranking"
        # experiment_sha = "gq_dm_test_ranking_el"
        # experiment_sha = "gq_dm_test"
        # experiment_sha = "gq_dm_test_el"

        # experiment_sha = "gq_seen_test_unconstrained"
        # experiment_sha = "gq_seen_test_unconstrained_el"
        # experiment_sha = "gq_seen_test_ranking"
        # experiment_sha = "gq_seen_test_ranking_el"
        # experiment_sha = "gq_seen_test"
        # experiment_sha = "gq_seen_test_el"

        # experiment_sha = "webqsp_el_scratch"

        # experiment_sha = "webqsp_zero_s2s"

        # experiment_sha = "gq_dm_val_ranking"
        # experiment_sha = "gq_seen_val_ranking"

        # experiment_sha = "gq1_preliminary_0927"

        experiment_sha = self._experiment_sha

        if same_logical_form(predicted_logical_form, target_logical_form):
            if self._eval:
                print(str(qid), self._get_utterance(source),
                      file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
                print("p: ", predicted_logical_form,
                      file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
                print("t: ", target_logical_form,
                      file=open("results/www/" + experiment_sha + ".correct.txt", "a"))
                # print(str(qid), self._get_utterance(source))
                # print("p: ", predicted_logical_form)
                # print("t: ", target_logical_form)
            return 1
        else:
            if self._eval:  # only save incorrect predictions to file
                print(str(qid), self._get_utterance(source),
                      file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
                print("p: ", predicted_logical_form,
                      file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
                print("t: ", target_logical_form,
                      file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
                denotation = []
                try:
                    sparql_query = lisp_to_sparql(predicted_logical_form)
                    denotation.extend(execute_query(sparql_query))
                except Exception:
                    pass
                print("a: ", '\t'.join(denotation),
                      file=open("results/www/" + experiment_sha + ".wrong.txt", "a"))
                # print(str(qid), self._get_utterance(source))
                # print("p: ", predicted_logical_form)
                # print("t: ", target_logical_form)
            return 0

    def _compute_F1(self, predicted: torch.Tensor,
                    constrained_vocab: List[str],
                    answer: List[str]):
        predicted_logical_form = self._get_logical_form(predicted, constrained_vocab)
        try:
            sparql_query = lisp_to_sparql(predicted_logical_form)
            denotation = set(execute_query(sparql_query))
            correct = denotation.intersection(set(answer))
            precision = len(correct) / len(denotation)
            recall = len(correct) / len(answer)

            return (2 * precision * recall) / (precision + recall)
        except Exception:
            return 0

    # def _compute_exact_match_k(self,
    #                            predicted_k: torch.Tensor,
    #                            target: torch.Tensor) -> int:
    #     target_logical_form = self._get_logical_form(target[1:])  # omit the start symbol
    #     for i, predicted in enumerate(predicted_k):
    #         predicted_logical_form = self._get_logical_form(predicted)
    #         if same_logical_form(predicted_logical_form, target_logical_form):
    #             return 1, 1. / (i + 1)
    #
    #     return 0, 0

    def _encode(self,
                # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
                embedded_input: torch.Tensor,
                schema_start: List[List[int]]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        # Here we only want to get the encoding for the utterance part, so we also mask the concatenated
        # schema constants out
        source_mask = self._get_utterance_mask_from_bert_input(schema_start, embedded_input.shape[0],
                                                               embedded_input.shape[1])
        if self.training:
            # The trick is to put this sentence right before calling the rnn
            # TODO: only use the first tokenized_source of the ListField as input
            self._encoder._module.flatten_parameters()

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    # get the mask for untterance in the concatenated input to BERT
    def _get_utterance_mask_from_bert_input(self, schema_start, batch_size, max_len):
        assert batch_size == len(schema_start)
        mask = torch.ones(batch_size, max_len)
        for i, l in enumerate(schema_start):
            mask[i, l[0]:] = 0
        try:
            return mask.to(self._device)
        except RuntimeError:
            mask = mask
            print(mask.shape)

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

    def _forward_loop(
            self,
            state: Dict[str, torch.Tensor],
            target_tokens: Dict[str, torch.LongTensor] = None
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

        batch_size = source_mask.size()[0]

        # shape: (batch_size, max_target_sequence_length)
        # targets = target_tokens.squeeze(-1)  # no need to do it again
        targets = target_tokens

        _, target_sequence_length = targets.size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif target_tokens is None:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # apply the vocab mask
            # output_projections.masked_fill_(vocab_mask == 1, -1e32)
            # pay attention whether 1 or 0 denotes mask (by convention it should be 0)
            output_projections.masked_fill_(vocab_mask == 0, -1e32)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # print(step_logits[0][0][0] != step_logits[0][0][0])
            # print((step_logits[0][0][0] != step_logits[0][0][0]).any())
            # print(list(map(int, list(step_logits[0][0][0] != step_logits[0][0][0]))).index(1))
            # print(numpy.where(numpy.array(list(step_logits[0][0][0] != step_logits[0][0][0])) == 1))

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

        if target_tokens is not None:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = (target_tokens != -1).long()
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
        # shape (last_predictions): (group_size, )  group_size can be batch
        # size or batch_size * beamsize or batch_size * num_of_candidates
        group_size = last_predictions.shape[0]
        batch_size = self._output_embedding.shape[0]
        assert group_size % batch_size == 0
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        if group_size == batch_size:
            # output_embedding: (batch_size, output_dim, num_classes)
            # (group_size, output_dim)
            embedded_input = self._output_embedding[torch.arange(batch_size).to(self._device), :,
                             last_predictions.clone().long()]
        else:
            index = torch.arange(batch_size).unsqueeze(1)
            index = index.repeat(1, group_size // batch_size)
            index = index.reshape(-1).to(self._device)

            embedded_input = self._output_embedding[index, :, last_predictions]

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

        # shape (decoder_hidden): (group_size, decoder_output_dim)
        # shape (decoder_context): (group_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
            decoder_input, (decoder_hidden, decoder_context)
        )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # (batch_size, decoder_output_dim, num_classes)
        output_embedding = self._output_embedding
        decoder_output_dim = output_embedding.shape[1]
        output_embedding = output_embedding.repeat(1, group_size // batch_size, 1)
        # (group_size, decoder_output_dim, num_classes)
        output_embedding = output_embedding.reshape(group_size, decoder_output_dim, -1)
        # (group_size, 1, decoder_output_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # (group_size, 1, num_classes)
        output_projections = torch.bmm(decoder_hidden, output_embedding)
        output_projections = output_projections.squeeze(1)

        return output_projections, state

    def _compute_target_embedding(self,
                                  schema_start: List[List[int]],
                                  schema_end: List[List[int]],
                                  bert_embeddings: torch.Tensor) -> torch.Tensor:

        #print('schema start')
        #print(schema_start)
        batch_size = len(schema_start)
        target_embedding = bert_embeddings.new_zeros([batch_size, self._vocab_size, 768])
        for i in range(batch_size):
            assert len(schema_start[i]) == len(schema_end[i])
            for j in range(len(schema_start[i])):
                assert j < target_embedding.shape[1]
                start = schema_start[i][j]
                end = schema_end[i][j]
                avg_embedding = bert_embeddings[i][j // self._num_constants_per_group][start: end + 1]
                avg_embedding = torch.sum(avg_embedding, dim=0)
                avg_embedding = avg_embedding / (end - start + 1)

                target_embedding[i][j] = avg_embedding


        # (batch_size, vocab_size, dim)
        return target_embedding

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
            logits: torch.LongTensor,
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
        # # TODO: comment this line of code and paste the error to pytorch forum
        # relevant_targets.masked_fill_(relevant_targets == -1, 0)
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
        all_metrics['F1'] = self._F1.get_metric(reset)
        all_metrics['exact_match_k'] = self._exact_match_k.get_metric(reset)
        all_metrics['MRR_k'] = self._MRR_k.get_metric(reset)
        return all_metrics

    # @overrides
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
        print('predicted_indices: ############')
        print(predicted_indices)
    
        constrained_vocab = output_dict["constrained_vocab"]
        print(constrained_vocab)
        ids = output_dict['ids']

        with open('output_vocab.pkl', 'wb') as file:
            pickle.dump(constrained_vocab, file)

        all_predicted_lfs = []
        all_predicted_answers = []
        for indices, constrained, qid in zip(predicted_indices, constrained_vocab, ids):
            if not self._ranking_mode:
                indices = indices[0]
            predicted_lf = self._get_logical_form(indices, constrained)

            #denotation = []
            #try:
            #    sparql_query = lisp_to_sparql(predicted_lf)
            #    #print('calling the sparql executor function')
            #    denotation.extend(execute_query(sparql_query))
            #except Exception:
            #    pass
            #all_predicted_answers.append(denotation)
            all_predicted_lfs.append(predicted_lf)

        rtn = {}
        rtn['qid'] = ids
        rtn['logical_form'] = all_predicted_lfs
        #rtn['answer'] = all_predicted_answers
        return rtn
