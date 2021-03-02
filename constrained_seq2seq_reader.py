import re
import numpy as np
from typing import Dict, Optional, List, Set
from collections import defaultdict
import logging
import json

from overrides import overrides

from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from entity_linking import surface_index_memory
from entity_linking.bert_entity_linker import BertEntityLinker
from entity_linking.value_extractor import GrailQA_Value_Extractor
from utils.search_over_graphs import generate_all_logical_forms_alpha, generate_all_logical_forms_2, \
    generate_all_logcial_forms_2_with_domain, get_vocab_info_online, generate_all_logical_forms_for_literal

logger = logging.getLogger(__name__)


@DatasetReader.register("cons_seq2seq")
class Constrained_Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``ComposedSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens : ``TextField`` and
        target_tokens : ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``SpacyTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : bool, (optional, default=True)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(
            self,
            source_tokenizer: Tokenizer = None,
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            source_add_start_token: bool = True,
            source_add_end_token: bool = True,
            delimiter: str = "\t",
            source_max_tokens: Optional[int] = None,
            target_max_tokens: Optional[int] = None,
            lazy: bool = False,
            offline: bool = True,
            training: bool = True,
            perfect_entity_linking: bool = True,
            constrained_vocab=None,
            ranking_mode: bool = False,  # need to be consistent with the model
            use_constrained_vocab: bool = False,  # need to be consistent with the model
            device: str = "cuda:0"   # device for entity linker, not semantic parser
    ) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or SpacyWordSplitter()
        self._target_tokenizer = target_tokenizer or (lambda x: x.replace('(', ' ( ').replace(')', ' ) ').split())
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self._training = training
        self._offline = offline
        self._perfect_el = perfect_entity_linking
        if not self._perfect_el:
            surface_index = surface_index_memory.EntitySurfaceIndexMemory(
                "entity_linking/data/entity_list_file_freebase_complete_all_mention", "entity_linking/data/surface_map_file_freebase_complete_all_mention",
                "freebase_complete_all_mention")
            self.linker = BertEntityLinker(surface_index, device=device)
            self.extractor = GrailQA_Value_Extractor()
        self._constrained_vocab = constrained_vocab or '1_step'
        # possible choices: {1_step, 2_step, cheating, domain， mix}
        self._ranking_mode = ranking_mode
        self._use_constrained_vocab = use_constrained_vocab

        self._uncovered_count = defaultdict(lambda: 0)

    @overrides
    def _read(self, file_path: str):
        # with open('cache/entity_names.json', 'r') as f:
        #     self._entity_names = json.load(f)

        if self._ranking_mode:
            with open('ontology/domain_info', 'r') as f:
                self._constants_to_domain = defaultdict(lambda: None)
                self._constants_to_domain.update(json.load(f))

        if self._use_constrained_vocab:
            if self._constrained_vocab == '1_step':
                with open('cache/1hop_vocab', 'r') as f:
                    self._vocab_info = json.load(f)

            if self._constrained_vocab == '2_step':
                with open('cache/2hop_vocab', 'r') as f:
                    self._vocab_info = json.load(f)

            if self._constrained_vocab == 'domain':
                with open('ontology/domain_dict', 'r') as f:
                    self._domain_dict = json.load(f)
                with open('ontology/domain_info', 'r') as f:
                    self._constants_to_domain = defaultdict(lambda: None)
                    self._constants_to_domain.update(json.load(f))

            if self._constrained_vocab in ['mix1', 'mix2']:
                if self._constrained_vocab == 'mix1':
                    with open('cache/1hop_vocab', 'r') as f:
                        self._vocab_info = json.load(f)
                elif self._constrained_vocab == 'mix2':
                    with open('cache/2hop_vocab', 'r') as f:
                        self._vocab_info = json.load(f)
                with open('ontology/domain_dict', 'r') as f:
                    self._domain_dict = json.load(f)
                with open('ontology/domain_info', 'r') as f:
                    self._constants_to_domain = defaultdict(lambda: None)
                    self._constants_to_domain.update(json.load(f))

        with open('ontology/domain_info', 'r') as f:
            self._schema_constants = set(json.load(f).keys())

        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            file_contents = json.load(data_file)
            for item in file_contents:
                if self._perfect_el:
                    entities = set()
                    entity_map = {}
                    for node in item['graph_query']['nodes']:
                        if node['node_type'] == 'entity':
                            entities.add(node['id'])
                            entity_map[node['id']] = ' '.join(
                                node['friendly_name'].split()[:5])
                    literals = set()
                    for node in item['graph_query']['nodes']:
                        if node['node_type'] == 'literal' and node['function'] not in ['argmin', 'argmax']:
                            literals.add(node['id'])
                else:
                    entity_map = self.linker.get_entities(item['question'])
                    entities = set(entity_map.keys())
                    for k in entity_map:
                        v = entity_map[k]
                        entity_map[k] = ' '.join(v.split()[:5])
                    literals = set()
                    mentions = self.extractor.detect_mentions(item['question'])
                    for m in mentions:
                        literals.add(self.extractor.process_literal(m))
                if self._ranking_mode:

                    logical_forms = []
                    if len(entities) > 0:
                        if self._perfect_el:
                            logical_forms.extend(generate_all_logical_forms_alpha(list(entities)[0],
                                                                                  offline=self._offline))  # use no domain info
                            logical_forms.extend(generate_all_logical_forms_2(list(entities)[0], offline=self._offline))
                        else:
                            for entity in entities:
                                logical_forms.extend(generate_all_logical_forms_alpha(entity, offline=self._offline))
                                logical_forms.extend(generate_all_logical_forms_2(entity, offline=self._offline))

                    for literal in literals:
                        logical_forms.extend(
                            generate_all_logical_forms_for_literal(literal))

                    if len(logical_forms) > 0:
                        yield self.text_to_instance(item, entity_map, literals, logical_forms)
                else:
                    new_instance = self.text_to_instance(item, entity_map, literals)
                    if new_instance:
                        yield new_instance

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(self,
                         item: Dict,
                         entity_map: Dict,
                         literals: Set,
                         logical_forms: List = None) -> Instance:  # type: ignore
        qid = MetadataField(item['qid'])

        source_string = item['question'].lower()
        if 's_expression' in item:
            target_string = item['s_expression']
        else:
            target_string = None
        values = []  # entities and literals in a question. Entities and values are not supposed to be put in vocab
        for e in entity_map:
            values.append(e)

        for k in entity_map:
            entity_map[k] = entity_map[k].lower()

        for l in literals:
            values.append(l)

        if len(values) == 0:
            values.append('placeholder')  # TODO: this is an expedient way to avoid the non-value bug

        if self._ranking_mode:
            lfs = []
            candidates_value_indices = []
            for lf in logical_forms:
                # The correct answer will be put in the first position later for training
                if lf == target_string and self._training:
                    continue
                tokenized_lf = [Token(x) for x in self._target_tokenizer(lf)]
                if self._target_max_tokens and len(tokenized_lf) > self._target_max_tokens:
                    self._target_max_exceeded += 1
                    tokenized_lf = tokenized_lf[: self._target_max_tokens]
                tokenized_lf.insert(0, Token(START_SYMBOL))
                tokenized_lf.append(Token(END_SYMBOL))
                lf_field = TextField(tokenized_lf, self._target_token_indexers)
                lfs.append(lf_field)
                candidate_value_indices = []
                for token in tokenized_lf:
                    try:  # token is literal or entity
                        candidate_value_indices.append(values.index(token.text))
                    except ValueError:
                        candidate_value_indices.append(-1)
                candidates_value_indices.append(ArrayField(np.array(candidate_value_indices), padding_value=-1))

            if self._training:
                tokenized_lf = [Token(x) for x in self._target_tokenizer(target_string)]
                if self._target_max_tokens and len(tokenized_lf) > self._target_max_tokens:
                    self._target_max_exceeded += 1
                    tokenized_lf = tokenized_lf[: self._target_max_tokens]
                tokenized_lf.insert(0, Token(START_SYMBOL))
                tokenized_lf.append(Token(END_SYMBOL))
                lf_field = TextField(tokenized_lf, self._target_token_indexers)

                candidate_value_indices = []
                lfs.insert(0, lf_field)  # for training, always put the correct answer in the first position
                for token in tokenized_lf:
                    try:  # token is literal or entity
                        candidate_value_indices.append(values.index(token.text))
                    except ValueError:
                        candidate_value_indices.append(-1)
                candidates_value_indices.insert(0, ArrayField(np.array(candidate_value_indices), padding_value=-1))

            candidates = ListField(lfs)
            candidates_value_indices = ListField(candidates_value_indices)

        # tokenized_source = self._source_tokenizer.tokenize(source_string)  # for bert
        # tokenized_source = [Token(x) for x in self._source_tokenizer(source_string)]
        tokenized_source = [x for x in self._source_tokenizer.split_words(source_string)]
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        if self._source_add_end_token:
            tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if self._use_constrained_vocab:
            if self._training:
                constrained_vocab = self._get_constrained_vocab(entity_map, item['s_expression'], item['domains'])
            else:
                constrained_vocab = self._get_constrained_vocab(entity_map)
        else:
            constrained_vocab = MetadataField(None)

        instance_dict = {"source_tokens": source_field,
                         "constrained_vocab": constrained_vocab,
                         "values": MetadataField(values),
                         "ids": qid,
                         "entity_name": MetadataField(entity_map)}

        if 'answer' in item:
            answer = []
            for a in item['answer']:
                answer.append(a['answer_argument'])
            instance_dict['answer'] = MetadataField(answer)

        if target_string is not None:
            tokenized_target = [Token(x) for x in self._target_tokenizer(target_string)]
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            # print(len(tokenized_target), len(set(tokenized_target).intersection(constrained_vocab)))
            # print(set(tokenized_target).difference(constrained_vocab))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            value_indices = []
            for token in target_field:
                try:  # token is literal or entity
                    value_indices.append(values.index(token.text))
                except ValueError:
                    value_indices.append(-1)

            instance_dict["value_indices"] = ArrayField(np.array(value_indices), padding_value=-1)

            def get_target_words(schema_item: str) -> List[str]:
                if schema_item in entity_map:
                    schema_item = entity_map[schema_item]
                    return re.split('[._ ]', schema_item)[:5]
                return re.split('[._ ]', schema_item)

            target_words = [Token(word) for x in self._target_tokenizer(target_string)
                            for word in get_target_words(x)]

            target_words.append(Token(START_SYMBOL))
            target_words.append(Token(END_SYMBOL))
            # The target words field is only used to construct the vocabulary for words in schema items
            target_words_field = TextField(target_words,
                                           {"tokens": SingleIdTokenIndexer(namespace='tgt_words')})
            instance_dict['target_tokens'] = target_field
            instance_dict['target_words'] = target_words_field
            if self._ranking_mode:
                instance_dict['candidates'] = candidates
                instance_dict['candidates_value_indices'] = candidates_value_indices
        else:
            if self._ranking_mode:
                instance_dict['candidates'] = candidates
                instance_dict['candidates_value_indices'] = candidates_value_indices

        return Instance(instance_dict)

    def _get_vocab_info(self, entity):
        if self._offline:
            if entity in self._vocab_info:
                return self._vocab_info[entity]
            else:
                return get_vocab_info_online(entity)
        else:
            return get_vocab_info_online(entity)

    def _get_constrained_vocab(self,
                               entity_map,
                               s_expression=None,
                               domains=None):
        if self._constrained_vocab in ['1_step', '2_step']:
            vocab = {'(', ')', 'JOIN', 'AND', 'R', 'ARGMAX', 'ARGMIN', 'COUNT', 'ge', 'gt', 'le', 'lt'}
            flag = False
            for e in entity_map:
                # vocab.update(self._get_vocab_info(e))
                vocab.update(set(self._get_vocab_info(e)).intersection(self._schema_constants))
                flag = True

            if not flag:  # no entity:
                vocab.update(self._schema_constants)

            vocab = [Token(x) for x in vocab]
            vocab.append(Token(END_SYMBOL))
            if self._training:
                vocab.extend([Token(x) for x in self._target_tokenizer(s_expression)])
                vocab = list(set(vocab))
            return TextField(vocab, self._target_token_indexers)
        elif self._constrained_vocab in ['mix1', 'mix2']:  # only used for ideal experiments
            assert self._training
            vocab = set()
            for domain in domains:
                vocab.update(self._domain_dict[domain])

            vocab_hop = set()
            flag = False
            for e in entity_map:
                # vocab_hop.update(self._get_vocab_info(e))
                vocab_hop.update(set(self._get_vocab_info(e)).intersection(self._schema_constants))
                flag = True

            if not flag:  # no entity:
                vocab.update(self._schema_constants)
            vocab = vocab.intersection(vocab_hop)

            vocab.update({'(', ')', 'JOIN', 'AND', 'R', 'ARGMAX', 'ARGMIN', 'COUNT', 'ge', 'gt', 'le', 'lt'})
            vocab = [Token(x) for x in vocab]
            vocab.append(Token(END_SYMBOL))
            if self._training:
                vocab.extend([Token(x) for x in self._target_tokenizer(s_expression)])
                vocab = list(set(vocab))
            return TextField(vocab, self._target_token_indexers)

        else:
            raise Exception('_constrained_vocab must be one of 1_step, 2_step, cheating, '
                            'but received {}'.format(self._constrained_vocab))


