import math 
"""
def createMatrix(n):
    matrix = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(0)
        matrix.append(t) 
    return matrix 

def computeProbabilities(train, init_table, word_tag_pairs, transition_matrix, tag_to_index, tags, tag_count, alpha):
    init_table = [init_table[i]/len(train) for i in range(len(init_table))]
    #compute smoothed probabilities given counts 
    for word in word_tag_pairs:
        for tag in word_tag_pairs[word]:
            word_tag_pairs[word][tag] = (alpha + word_tag_pairs[word][tag]) / (tags[tag] + alpha*(1 + tag_count))
            #print("("+word+","+tag+"): " + str(word_tag_pairs[word][tag]))
    for tag, ct in tags.items():
        curr = tag_to_index[tag]
        for nxt in range(tag_count):
            transition_matrix[curr][nxt] = (alpha + transition_matrix[curr][nxt]) / (ct + alpha*(1 + tag_count))
    return init_table, word_tag_pairs, transition_matrix
    
def buildTrainingModel(train, smoothing_param):
    tags = {}           #tracks the total occurance of a specific tag 
    tag_to_index = {}   #stores a tag and its index...{START: 0, ADJ: 1, etc...}
    index_to_tag = {}   #stores a tag and its index...{0: START, 1: ADJ, etc...}
    tag_count = 0       #unique tags     
    word_tag_pairs = {} #tracks the occurance of a {WORD1: {START: 0, ADJ: 1,..}}
    #build word tag pair counts, and tag counts 
    for sentence in train:
        for word, tag in sentence: 
            if tag not in tags: #update the tag_indices and tag_count for every new tag encoutnered 
                tags[tag] = 1
                tag_to_index[tag] = tag_count
                index_to_tag[tag_count] = tag 
                tag_count += 1
            else:
                tags[tag] += 1
            if word not in word_tag_pairs:
                word_tag_pairs[word] = {}
                word_tag_pairs[word][tag] = 1
            else:
                if tag not in word_tag_pairs[word]:
                    word_tag_pairs[word][tag] = 1
                else:
                    word_tag_pairs[word][tag] += 1
    transition_matrix = createMatrix(tag_count)
    init_table = [0 for i in range(tag_count)]   
    for sentence in train: 
        for i, (word, tag) in enumerate(sentence[:-1]):
            if i == 0:
                init_table[tag_to_index[tag]] += 1
            transition_matrix[tag_to_index[tag]][tag_to_index[sentence[i+1][1]]] += 1
    in_probabilities, wt_probabilities, tr_probabilities = computeProbabilities(train, init_table, word_tag_pairs, transition_matrix, tag_to_index, tags, tag_count, smoothing_param)
    return wt_probabilities, tr_probabilities, in_probabilities, tags, tag_to_index, index_to_tag, tag_count

def runTrellis(test, wt_probabilities, tr_probabilities, in_probabilities, tags, tag_to_index, index_to_tag, tag_count, smoothing_param):
    predictions = []
    for sentence in test:
        trellis = [[(0, (0,0)) for i in range(len(sentence))] for j in range(tag_count)] 
        for i, word in enumerate(sentence):
            for tag, j in tag_to_index.items(): 
                if i == 0:
                    
    return predictions
def viterbi_1(train, test):
    smoothing_param = 0.001
    wt_probabilities, tr_probabilities, in_probabilities, tags, tag_to_index, index_to_tag, tag_count = buildTrainingModel(train, smoothing_param)
    return runTrellis(test, wt_probabilities, tr_probabilities, in_probabilities, tags, tag_to_index, index_to_tag, tag_count, smoothing_param)
"""