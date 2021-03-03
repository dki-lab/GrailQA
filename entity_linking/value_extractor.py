import re
import json
from collections import defaultdict


class GrailQA_Value_Extractor:
    def __init__(self):
        self.pattern = r"(?:\d{4}-\d{2}-\d{2}t[\d:z-]+|(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}|\d{2}/\d{4}|[-]*\d+[.]\d+e[+-]\d+|[-]*\d+e[+-]\d+|(?<= )[-]*\d+[.]\d+|^[-]*\d+[.]\d+|(?<= )[-]*\d+|^[-]*\d+)"

    def detect_mentions(self, question):
        return re.findall(self.pattern, question)

    def process_literal(self, value: str):  # process datetime mention; append data type
        pattern_date = r"(?:(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
        pattern_datetime = r"\d{4}-\d{2}-\d{2}t[\d:z-]+"
        pattern_float = r"(?:[-]*\d+[.]*\d*e[+-]\d+|(?<= )[-]*\d+[.]\d*|^[-]*\d+[.]\d*)"
        pattern_yearmonth = r"\d{4}-\d{2}"
        pattern_year = r"(?:(?<= )\d{4}|^\d{4})"
        pattern_int = r"(?:(?<= )[-]*\d+|^[-]*\d+)"

        if len(re.findall(pattern_datetime, value)) == 1:
            value = value.replace('t', "T").replace('z', 'Z')
            return f'{value}^^http://www.w3.org/2001/XMLSchema#dateTime'
        elif len(re.findall(pattern_date, value)) == 1:
            if value.__contains__('-'):
                return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
            elif value.__contains__('/'):
                fields = value.split('/')
                value = f"{fields[2]}-{fields[0]}-{fields[1]}"
                return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
            else:
                if value.__contains__('jan.'):
                    month = '01'
                elif value.__contains__('feb.'):
                    month = '02'
                elif value.__contains__('mar.'):
                    month = '03'
                elif value.__contains__('apr.'):
                    month = '04'
                elif value.__contains__('may'):
                    month = '05'
                elif value.__contains__('jun.'):
                    month = '06'
                elif value.__contains__('jul.'):
                    month = '07'
                elif value.__contains__('aug.'):
                    month = '08'
                elif value.__contains__('sep.'):
                    month = '09'
                elif value.__contains__('oct.'):
                    month = '10'
                elif value.__contains__('nov.'):
                    month = '11'
                elif value.__contains__('dec.'):
                    month = '12'
                pattern = "(?<=the )\d+"
                day = re.findall(pattern, value)[0]
                if len(day) == 1:
                    day = f"0{day}"
                pattern = "(?<=, )\d+"
                year = re.findall(pattern, value)[0]
                return f'{year}-{month}-{day}^^http://www.w3.org/2001/XMLSchema#date'
        elif len(re.findall(pattern_yearmonth, value)) == 1:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#gYearMonth'
        elif len(re.findall(pattern_float, value)) == 1:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#float'
        elif len(re.findall(pattern_year, value)) == 1 and int(value) <= 2015:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#gYear'
        else:
            return f'{value}^^http://www.w3.org/2001/XMLSchema#integer'


if __name__ == '__main__':
    with open('../data/grailqa_v1.0_test.json') as f:
        data = json.load(f)

    extractor = GrailQA_Value_Extractor()

    data_types = defaultdict(lambda: set())
    count = 0
    f1_sum = 0
    recall_sum = 0
    for item in data:
        flag = False
        literals = []
        for node in item['graph_query']['nodes']:
            if node['node_type'] == 'literal' and not node['function'] in ['argmin', 'argmax']:
                flag = True
                # literals.append(node['id'].split('^^')[0])
                literals.append(node['id'])
                data_types[node['id'].split('^^')[1]].add(node['id'])
                # if node['id'].__contains__('gYear'):
                #     print(item['sparql_query'])
        if flag:
            # print(item['question'])
            # print(item['s_expression'])
            # print("gold:", literals)
            count += 1
            predicted = []
            mentions = extractor.detect_mentions(item['question'])
            for m in mentions:
                predicted.append(extractor.process_literal(m))
            # print(predicted)
            literals = set(literals)
            predicted = set(predicted)
            if len(predicted) > 0:
                precision = len(predicted.intersection(literals)) / len(predicted)
            else:
                precision = 0
            recall = len(predicted.intersection(literals)) / len(literals)
            if recall != 0 and precision != 0:
                f1 = 2 / (1 / precision + 1 / recall)
            else:
                f1 = 0
            f1_sum += f1
            recall_sum += recall
            print(f1)
            if f1 != 1:
                print(item['question'])
                print(literals)
                print(predicted)

    for k in data_types:
        print(k)
        print(data_types[k])

    print(count, f1_sum, recall_sum)