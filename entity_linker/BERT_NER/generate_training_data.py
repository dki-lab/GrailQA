import json
import random
from nltk import word_tokenize


with open('../data/grailqa_v1.0_train.json') as f:
    data = json.load(f)
# with open('../data/webqsp_0107.train.json') as f:
#     data = json.load(f)

annotations = []
for item in data:
    entity_mentions = []
    for node in item['graph_query']['nodes']:
        if node['node_type'] == 'entity' and ("implicit" not in node or not node["implicit"]):
            entity_mentions.append(word_tokenize(node['friendly_name'].lower()))
    if len(entity_mentions) > 0:
        utterance = word_tokenize(item['question'])
        # print(entity_mentions)
        # print(word_tokenize(item['question']))
        tags = ['O' for i in range(len(utterance))]
        for mention in entity_mentions:
            l = len(mention)
            for i in range(len(utterance) - l + 1):
                if utterance[i: i+l] == mention:
                    tags[i] = 'B'
                    tags[i+1: i+l] = ['I' for i in range(l - 1)]

        annotations.append({"text": utterance, "tags": tags})

random.shuffle(annotations)
print(len(annotations))
# json.dump(annotations, open("BERT_NER/train_webq_0115.json", 'w'))
json.dump(annotations, open("BERT_NER/grail_train.json", 'w'))

