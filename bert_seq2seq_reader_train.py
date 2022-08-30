
# %%
import re
from typing import Dict, Optional, List, Set
from collections import defaultdict
import logging
import json
import random

from overrides import overrides

from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from entity_linking.value_extractor import GrailQA_Value_Extractor
from utils.search_over_graphs import generate_all_logical_forms_alpha, generate_all_logical_forms_2, \
    get_vocab_info_online, generate_all_logical_forms_for_literal
from utils import search_over_graphs

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_seq2seq")
class Bert_Seq2SeqDatasetReader(DatasetReader):
    def __init__(
            self,
            source_tokenizer: Tokenizer = PretrainedTransformerTokenizer(
                model_name="bert-base-uncased",
                do_lowercase=True
            ),
            target_tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = PretrainedTransformerIndexer(
                model_name="bert-base-uncased",
                do_lowercase=True,
                namespace='bert'
            ),
            target_token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = False,
            offline: bool = True,
            training: bool = True,
            use_constrained_vocab: bool = True,
            constrained_vocab=None,
            ranking_mode: bool = False,
            perfect_entity_linking: bool = True,
            source_max_tokens=512,
            num_constants_per_group=30,
            delimiter=";",
            gq1=False,
            use_sparql=False  # whether to use sparql as target logical form. Using S-expression by default
    ) -> None:
        super().__init__(lazy)
        search_over_graphs.gq1 = gq1
        self._gq1 = gq1
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer or (lambda x: x.replace('(', ' ( ').replace(')', ' ) ').split())
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self._training = training
        self._ranking_mode = ranking_mode
        self._offline = offline
        self._use_constrained_vocab = use_constrained_vocab
        self._constrained_vocab = constrained_vocab or '1_step'
        # possible choices: {1_step, 2_step, cheating, domain， mix1, mix2}
        self._source_max_tokens = source_max_tokens

        self._use_sparql = use_sparql
        if self._use_sparql:
            self._target_tokenizer = target_tokenizer or (
                lambda x: x.replace('(', ' ( ').replace(')', ' ) ').replace('http:', 'http.').replace(':',
                                                                                                      ' : ').replace(
                    'http.', 'http:').split())
        if not self._use_sparql:
            self._global_syntax_constants_vocab = {'(': 0, ')': 1, 'JOIN': 2, 'AND': 3, 'R': 4, 'ARGMAX': 5,
                                                   'ARGMIN': 6,
                                                   'COUNT': 7, 'ge': 8, 'gt': 9, 'le': 10, 'lt': 11,
                                                   START_SYMBOL: 12, END_SYMBOL: 13}
            # "@@PADDING@@": 13}
        else:
            self._global_syntax_constants_vocab = {'(': 0, ')': 1, '!=': 2, '&&': 3, ':': 4, '<': 5,
                                                   '<=': 6, '>': 7, '>=': 8, '?value': 9, '?x0': 10, '?x1': 11,
                                                   '?x2': 12, '?x3': 13, '?y0': 14, '?y1': 15, '?y2': 16, '?y3': 17,
                                                   'AS': 18, 'COUNT': 19, 'DISTINCT': 20, 'FILTER': 21, 'MAX': 22,
                                                   'MIN': 23, 'PREFIX': 24, 'SELECT': 25, 'VALUES': 26, 'WHERE': 27,
                                                   'rdf': 28, 'rdfs': 29, '{': 30, '}': 31, '.': 32,
                                                   '<http://www.w3.org/1999/02/22-rdf-syntax-ns#>': 33,
                                                   '<http://www.w3.org/2000/01/rdf-schema#>': 34,
                                                   '<http://rdf.freebase.com/ns/>': 35,
                                                   'type.object.type': 36,
                                                   START_SYMBOL: 37, END_SYMBOL: 38}

        self._perfect_el = perfect_entity_linking
        if not self._perfect_el:
            el_fn = "graphq_el.json" if self._gq1 else "grailqa_el.json"
            self.el_results = json.load(open("entity_linking/" + el_fn))
            self.extractor = GrailQA_Value_Extractor()
        self._num_constants_per_group = num_constants_per_group
        self._delimiter = delimiter

    @overrides
    def _read(self, file_path: str):
        #if self._ranking_mode:
        #    with open('ontology/domain_info', 'r') as f:
        #        self._constants_to_domain = defaultdict(lambda: None)
        #        self._constants_to_domain.update(json.load(f))

        if self._gq1:
            suffix = "_gq1"
        else:
            suffix = ""
        #if self._use_constrained_vocab:
        #    if self._constrained_vocab == '1_step':
        #        with open(f'cache/1hop_vocab{suffix}', 'r') as f:
        #            self._vocab_info = json.load(f)###

        #    if self._constrained_vocab == '2_step':
        #        with open(f'cache/2hop_vocab{suffix}', 'r') as f:
        #            self._vocab_info = json.load(f)

        #    if self._constrained_vocab == 'domain':
        #        with open('ontology/domain_dict', 'r') as f:
        #            self._domain_dict = json.load(f)
        #        with open('ontology/domain_info', 'r') as f:
        #            self._constants_to_domain = defaultdict(lambda: None)
        #            self._constants_to_domain.update(json.load(f))

        #    if self._constrained_vocab in ['mix1', 'mix2']:
        #        if self._constrained_vocab == 'mix1':
        #            with open(f'cache/1hop_vocab{suffix}', 'r') as f:
        #                self._vocab_info = json.load(f)
        #        elif self._constrained_vocab == 'mix2':
        #            with open(f'cache/2hop_vocab{suffix}', 'r') as f:
        #                self._vocab_info = json.load(f)
        #        with open('ontology/domain_dict', 'r') as f:
        #            self._domain_dict = json.load(f)
        #        with open('ontology/domain_info', 'r') as f:
        #            self._constants_to_domain = defaultdict(lambda: None)
        #            self._constants_to_domain.update(json.load(f))

        # if not self._training:
        #with open('ontology/domain_info', 'r') as f:
        #    self._schema_constants = set(json.load(f).keys())

        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            file_contents = json.load(data_file)
            for item in file_contents:
                #if self._perfect_el:
                #    entities = set()
                #    entity_map = {}
                #    for node in item['graph_query']['nodes']:
                #        if node['node_type'] == 'entity':
                #            entities.add(node['id'])
                ##            entity_map[node['id']] = ' '.join(
                 #               node['friendly_name'].replace(self._delimiter, ' ').split()[:5])
                #    literals = set()
                #    for node in item['graph_query']['nodes']:
                #        if node['node_type'] == 'literal' and node['function'] not in ['argmin', 'argmax']:
                #            literals.add(node['id'])
                #else:
                    # entity_map = self.linker.get_entities(item['question'])
                    #print(type(item['qid']))
                    #raise AttributeError
                 #   entity_map = self.el_results[str(item['qid'])]['entities']
                 #   entities = set(entity_map.keys())
                 #   for k in entity_map:
                 #       v = entity_map[k]['friendly_name']
                 #       entity_map[k] = ' '.join(v.replace(self._delimiter, ' ').split()[:5])
                 #   # print("linked entities:", entities)
                 #   literals = set()
                 #   mentions = self.extractor.detect_mentions(item['question'])
                 #   for m in mentions:
                 #       literals.add(self.extractor.process_literal(m))
                if not self._ranking_mode:
                    #dummies for smoke testing
                    entity_map = []
                    literals = []
                    instance = self.text_to_instance(item, entity_map, literals)
                else:
                    # domains = set()
                    # for edge in item['graph_query']['edges']:
                    #     if self._constants_to_domain[edge['relation']]:
                    #         domains.add(self._constants_to_domain[edge['relation']])

                    logical_forms = []
                    #if len(entities) > 0:
                    #    if self._perfect_el:
                    #        # logical_forms = generate_all_logical_forms_alpha(list(entities), list(domains), offline=False)
                    #        logical_forms.extend(generate_all_logical_forms_alpha(list(entities)[0],
                    #                                                              offline=self._offline))  # use no domain info
                    #        lfs_2 = generate_all_logical_forms_2(list(entities)[0], offline=self._offline)
                    #        if len(lfs_2) < 10000 or not self._gq1:
                    #            logical_forms.extend(lfs_2)
                    #        # logical_forms.extend(
                    #        # generate_all_logcial_forms_2_with_domain(list(entities)[0], list(domains)[0], offline=False))
                    #        # logical_forms = generate_all_logcial_forms_2_with_domain(list(entities)[0], list(domains)[0])
                    #        # logical_forms = generate_all_logical_forms_2(list(entities)[0])
                    #    else:
                    ##        for entity in entities:
                     #           logical_forms.extend(generate_all_logical_forms_alpha(entity, offline=self._offline))
                     #           lfs_2 = generate_all_logical_forms_2(list(entities)[0], offline=self._offline)
                     #           if len(lfs_2) < 10000 or not self._gq1:
                     #               logical_forms.extend(lfs_2)

                    #for literal in literals:
                    #    logical_forms.extend(
                    #        generate_all_logical_forms_for_literal(literal))

                    #if len(logical_forms) == 0:
                    #    continue

                    # print(len(logical_forms))
                    #instance = self.text_to_instance(item, entity_map, literals, logical_forms)
                #print('attempting to yield')
                if instance:
                    #print('yielding instance')
                    yield instance

    @overrides
    def text_to_instance(self,
                         item: Dict,
                         entity_map: Dict = [],
                         literals: Set = [],
                         logical_forms: List = None) -> Instance:  # type: ignore
        """
        This function takes the dictionary entity map as input and returns an AllenNLP Instance object that has the prepared sets of tokens to be fed to BERT, I think. Vocab item finding handled with 
        get_constrained_vocab function
        """
        
        ##print('SAVE VALUE: item')
        #print(item)
        #print('SAVE VALLUE: Entity map')
        #print(entity_map)
        #print(item)
        wikiqid = item['wikidata_qids']
       
        #print(qid)

        constrained_vocab = []
        question_num = item['qid']
       #print(f'reading question {question_num}')

        try:
            with open(f'cache/finetune/one_neighbor/finetune_biosparql_oneneighbor_cleaned_{wikiqid}.json', 'r') as file:
            #with open(f'cache/finetune/predicates/finetune_biosparql_predicates_cleaned_{wikiqid}.json', 'r') as file:
                vocab_dict = json.load(file)
                constrained_vocab.extend(vocab_dict[wikiqid])
        except FileNotFoundError:
            #raise AssertionError
            return None

        print('question: ', question_num)
        
       # print(len(constrained_vocab))
        target_string = item['s_expression']
        qid = MetadataField(question_num)
        #if item['qid'] in [2102902009000]:   # will exceed maximum length constraint
        #    return None

        #if not self._use_sparql:
        #    if 's_expression' in item:
        #        target_string = item['s_expression']
        #    else:
        #        target_string = None
        #else:
        #    if 'sparql_query' in item:
        #        target_string = item['sparql_query']
        #    else:
        #        target_string = None
        #item['question'] = item['question'].replace(self._delimiter, ' ')
        ## if self._training:
        #if self._use_constrained_vocab and len(entity_map) > 0:
        #    if not self._training:
        #        constrained_vocab = self._get_constrained_vocab(entity_map, literals)
        #    else:
        #        logical_form = item['s_expression'] if not self._use_sparql else item['sparql_query']
        #        domains = item['domains'] if not self._gq1 else None
        #        constrained_vocab = self._get_constrained_vocab(entity_map, literals, s_expression=logical_form,
        #                                                        domains=domains)
        #elif len(entity_map) == 0 and self._training:
        #    vocab = set()
        #    vocab.update(self._schema_constants)
        #    vocab = list(vocab)
        #    random.shuffle(vocab)
        #    vocab = set(vocab[:200])
        #    if not self._use_sparql:
        #        vocab.update([x for x in self._target_tokenizer(item['s_expression'])])
        #    else:
        #        vocab.update([x for x in self._target_tokenizer(item['sparql_query'])])


        #else:
        #    #print('entity map length')
        #    #print(len(entity_map))
        #    vocab = set()
        #    vocab.update(self._schema_constants)
        #    for eid in entity_map:
        #        vocab.add(eid)#

         #   for l in literals:
         #       vocab.add(l)

            #print('literals')
            #print(literals)
            #print('vocab')
            #print(vocab)
            #print('example of case with no entities')
            #raise AssertionError

            #constrained_vocab = list(vocab)
        #print('SAVE VALUE: Raw vocabulary')
        #print(constrained_vocab)
        # schema_constants = constrained_vocab[:]
        # always fix the position of END_SYMBOL, START_SYMBOL in each constrained vocab,
        # because a consistent global shared end_index / start_index is needed by BeamSearch
        # Here we also fix the position for all other syntactic constants for the convenience
        # of embeddings computing
        for k, v in {k: v for k, v in sorted(self._global_syntax_constants_vocab.items(), key=lambda x: x[1])}.items():
            constrained_vocab.insert(v, k)

        schema_constants = constrained_vocab[:]

        # dividing the schema constants into num_constants_per_group every group
        concat_strings = ['' for _ in range(len(schema_constants) // self._num_constants_per_group + 1)]
        for i in range(len(schema_constants) // self._num_constants_per_group + 1):
            if (i + 1) * self._num_constants_per_group <= len(schema_constants):
                right_index = (i + 1) * self._num_constants_per_group
            else:
                right_index = len(schema_constants)
            for constant in schema_constants[i * self._num_constants_per_group: right_index]:
                #if constant in entity_map:  # to get the representation for a entity based on its friendly name
                #    constant = entity_map[constant]
                if constant == '.':  # '.' in sparql means and
                    constant = 'and'
                concat_strings[i] += ' '.join(re.split('\.|_', constant.lower())) + self._delimiter
       # print('concat strings')
       # print('concat strings length: ', sum([len(string) for string in concat_strings]))
       # print(concat_strings)
        # handle sequence of length > 512 (dividing the schema constants into num_constants_per_group every group)
        # _source_tokenizer.tokenize will append the head [CLS] and ending [SEP] by itself
        tokenized_sources = [self._source_tokenizer.tokenize(item['question'] + '[SEP]' + concat_string)
                             for concat_string in concat_strings]

        #print('SAVE VALUE: Tokenized vocab')
        #print(tokenized_sources)
        #print('delimiter', self._delimiter)
        end = []
        start = []
        for tokenized_source in tokenized_sources:
            flag = False
            for i, token in enumerate(tokenized_source):
                if flag and str(token) == self._delimiter:
                    end.append(i - 1)
                    start.append(i + 1)
                if str(token) == '[SEP]':
                    if not flag:
                        #print('appending to start only')
                        start.append(i + 1)
                    flag = True

            start = start[:-1]  # ignore the last ';'

        # unit test for concatenation
        # print(len(constrained_vocab), constrained_vocab)
        # for i, tokenized_source in enumerate(tokenized_sources):
        #     print(constrained_vocab[14 + 50*i: 14 + min(50*(i + 1), len(start))])
        #     print(start[50*i:min(50*(i + 1), len(start))])
        #     print(end[50*i:min(50*(i + 1), len(start))])
        #     print(tokenized_source)

        # source_field = ListField(
        # [TextField(tokenized_source, self._source_token_indexers) for tokenized_source in tokenized_sources])

        source_field = []
        for tokenized_source in tokenized_sources:
            chunk = TextField(tokenized_source, self._source_token_indexers)
            if len(chunk) > self._source_max_tokens:
                print(len(chunk), item['qid'], '!!!!!!!!!')
                exit(-1)
            source_field.append(chunk)
            #print('source_field')
            #print(source_field)
        source_field = ListField(source_field)

        # vocab_field = TextField([Token(x) for x in constrained_vocab], self._target_token_indexers)
        vocab_field = MetadataField(constrained_vocab)
        # if len(constrained_vocab) != 14 + len(start):
        if len(constrained_vocab) != len(start):
            print(entity_map)
            print(len(constrained_vocab))
            print(len(start))
            print(len(end))
            print(start)
        # assert len(constrained_vocab) == 14 + len(start)
        assert len(constrained_vocab) == len(start)

        instance_dict = {"source_tokens": source_field,  # The concatenation of utterance and schema constants
                         # The start position for each schema constant in the concatenated input.
                         "schema_start": MetadataField(start),
                         # The end position ...
                         "schema_end": MetadataField(end),
                         "constrained_vocab": vocab_field,
                         "ids": qid}

        # If you want to use F1 during training, uncomment this!
        # if 'answer' in item:
        #     answer = []
        #     for a in item['answer']:
        #         answer.append(a['answer_argument'])
        #     instance_dict['answer'] = MetadataField(answer)

        # print("num lfs: ", len(logical_forms))
        if not self._training and self._ranking_mode and logical_forms:
            lfs = []
            for lf in logical_forms:
                try:
                    lf_field = self._convert_target_to_indices(lf, constrained_vocab, vocab_field)
                    if lf_field is None:
                        return None
                    else:
                        lfs.append(lf_field)
                except Exception:
                    pass
            if len(lfs) == 0:
                return None
            candidates = ListField(lfs)
            instance_dict["candidates"] = candidates

            #print(len(candidates))

        if target_string is not None:
            target_field = self._convert_target_to_indices(target_string, constrained_vocab, vocab_field)
            if target_field is None:
                return None
            else:
                instance_dict["target_tokens"] = target_field  # The id of each target token in constrained_vocab
        
        #print(instance_dict.keys())

        return Instance(instance_dict)

    def _convert_target_to_indices(self, target_string: str, constrained_vocab, vocab_field):
        converted_target = []
        for x in self._target_tokenizer(target_string):
            try:
                converted_target.append(constrained_vocab.index(x))
            except Exception:
                #assert not self._training
                # This would never happen during training. It only happens when a target token falls out of the vocab.
                # It may have some minor effect on loss during validation, no big deal
                #converted_target.append(0)
                return None

        converted_target.append(constrained_vocab.index(END_SYMBOL))
        converted_target.insert(0, constrained_vocab.index(START_SYMBOL))
        
        #print('converted_target')
        #print(converted_target)
        targets = []
        for t in converted_target:
            targets.append(IndexField(t, vocab_field))
        target_field = ListField(targets)
        return target_field

    def _get_vocab_info(self, entity, step=2):
        if self._offline:
            if entity in self._vocab_info:
                return self._vocab_info[entity]
            else:
                return get_vocab_info_online(entity, step=step)
        else:
            return get_vocab_info_online(entity, step=step)

    def _get_constrained_vocab(self,
                               entity_map,
                               literals,
                               s_expression=None,
                               domains=None):
        '''
        This function assembles a list of schema 'vocabulary' items that are within self._constrained_vocab steps of the entities in entity map. I believe entity map is a dictionary of entities identified by the entity linker
        and the vocab list is assembled from the files of 2step relations. 

        :param item:
        :param constrained_vocab:
        :return: Here this method only returns schema constants but not syntactic constants like AND, JOIN,...
        '''

        ## BP messing with this
        #print(self._constrained_vocab)
        #print('entity map')
        #print(type(entity_map))
        #print(entity_map)
        if self._constrained_vocab in ['1_step', '2_step']:
            # vocab = {'(', ')', 'JOIN', 'AND', 'R', 'ARGMAX', 'ARGMIN', 'COUNT', 'ge', 'gt', 'le', 'lt'}
            vocab = set()
            vocab.add('common.topic')  # for webquestions
            if self._constrained_vocab == '1_step':
                step = 1
            else:
                step = 2
            for e in entity_map:
                vocab.update(self._get_vocab_info(e, step))
                vocab.add(e)
            for l in literals:
                vocab.add(l)

            if self._training:
                vocab = set(list(vocab)[:1000])    # to reduce memory consumption
                vocab.update([x for x in self._target_tokenizer(s_expression)])
            vocab = list(vocab)
            # random.shuffle(vocab)
        elif self._constrained_vocab in ['mix1', 'mix2']:  # only used for ideal experiments
            assert self._training
            vocab = set()

            if self._constrained_vocab == 'mix1':
                step = 1
            else:
                step = 2

            for domain in domains:
                if domain != 'common' and domain != None:
                    vocab.update(self._domain_dict[domain])

            vocab_hop = set()
            for e in entity_map:
                vocab_hop.update(self._get_vocab_info(e, step))
                vocab_hop.add(e)
            for l in literals:
                vocab_hop.add(l)

            vocab = vocab.intersection(vocab_hop)

            vocab.update([x for x in self._target_tokenizer(s_expression)])
            vocab = list(vocab)
            # random.shuffle(vocab)
        else:
            raise Exception('_constrained_vocab must be one of 1_step, 2_step, cheating, '
                            'but received {}'.format(self._constrained_vocab))

        for syntax_constant in self._global_syntax_constants_vocab:
            if syntax_constant in vocab:
                vocab.remove(syntax_constant)
        #print('constrained vocab')
        #print(type(vocab))
        #print(vocab)
        return vocab

