### Data Format

Each file contains an array of JSON objects, each corresponding to a question (not graph query). The detailed format of a question is as follows. Note that for the public test set, we mask all the fields except for the input natrual language question (i.e., question field).

- qid: question id. The first 8 digits are for cannonical logical forms (i.e., templates), the remaining digits are for paraphrases and different groudings.. The questions with the same graph query id therefore are paraphrases. Note that, qids in test set are anonymized.
- question: the natural language question. Lower-cased
- answer: the answer set, an array of json objects. We provide answers in both human readable form and Freebase specific form like mid.
    - answer_type: The type of answer, can be either "Entity" or "Value"
    - answer_argument: The mid of an entity if AnswerType is "Entity", or the value itself if answer type is "Value"
    -entity_name: The friendly name (i.e., label) of the entity if AnswerType is "Entity".
- function: the function of a question, ["count", "max", "min", "argmax", "argmin", ">", ">=", "<", "<=", "none"]. A question will have at most one function
- num_node: number of nodes of the corresponding graph query
- num_edge: number of edges of the corresponding graph query
- graph_query: the corresponding graph query
    - nodes: the nodes, an array of JSON objects 
        - nid: node id, starting from 0
        - node_type: ["class", "entity", "literal"]. "class" node is either the question node or an ungrounded node, while "entity" and "literal" nodes are grounded
        - id: the Freebase unique ID of the node. For a class, it's the class name; for an entity, it's the mid; fora literal, it's the lexical form along with the data type
        - friendly_name: the canonical name of the node from Freebase, only for human readability
        - question_node: whether the node is the question node, [1, 0]. Actually it's useless for now because we currently only have one question node, which is always node 0
        - function: the function applied on the node
    - edges: the edges, an array of JSON objects
        - start: the node id of the starting node
        - end: the node id of the ending node
        - relation: the Freebase id of the relation on the edge
        - friendly_name: the Freebase canonical name of the relation, only for human readability
- sparql_query: the SPARQL query that is actually used to generate the answer. Note that the provided query will only get the Freebase id of the answer, and you need to convert it into the human readable format as described previously
- s_expression: the logical form in S-expression as described in our paper. It provides more concise syntax than sparql_query and can be easily used with modern encoder-decoder models. Note that, **if you want to have exact match score of your model on the leaderboard, please submit your results with logical form in S-expression.**
