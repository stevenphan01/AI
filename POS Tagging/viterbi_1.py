import math 

def viterbi_1(train, test):
    alpha = 1e-5 #emission
    beta  = 1e-5 #transition
    gamma = 1e-5 #init 
    tag_list, total_tags, transition, init_tags, word_tag = buildDataSet(train)
    v_words = len(word_tag)   #unique words
    v_init  = len(train)      #number of start tags
    v_tags  = len(total_tags) #number of unique tags 
    return trellis(test, tag_list, total_tags, transition, init_tags, word_tag, v_words, v_init, v_tags, alpha, beta, gamma) 

def buildDataSet(train):
    tag_list     = [] #                  [START, NOUN, ADJ...]
    total_tags   = {} #total tag counts  {START: #, NOUN: #, ADJ: #, etc...}
    transition   = {} #transitions       {(prev_tag, curr_tag): #, (prev_tag, curr_tag): #, etc...}
    init_tags    = {} #initial tags      {START: #, NOUN: #, ADJ: #, etc..}
    word_tag     = {} #emission          {(WORD, TAG): COUNT, (WORD, TAG): COUNT, (WORD, TAG): COUNT}
    for sentence in train:
        init_tag = sentence[0][1]
        if init_tag not in init_tags:
            init_tags[init_tag] = 0
        init_tags[init_tag] += 1
        for word, tag in sentence:
            if (word, tag) not in word_tag:
                word_tag[(word,tag)] = 0
            if tag not in total_tags:
                total_tags[tag] = 0
                tag_list.append(tag) #update tag_list everytime a new tag is encountered  
            word_tag[(word,tag)] += 1
            total_tags[tag] += 1
        #build transitions, starting at second word in the sentence
        for i in range(1, len(sentence)):
            curr_tag = sentence[i][1]
            prev_tag = sentence[i-1][1]
            key = (prev_tag, curr_tag)
            if key not in transition:
                transition[key] = 1
            else:
                transition[key] += 1
    return tag_list, total_tags, transition, init_tags, word_tag

def trellis(test, tag_list, total_tags, transition, init_tags, word_tag, v_words, v_init, v_tags, alpha, beta, gamma):
    predictions = []
    start = -1 #start sentinal used to backtrack 
    #trellis needs to store pointer to previous node and best probability: represented by a node (pointer, probability), pointer will be a coordinate (x,y)
    for i, sentence in enumerate(test):
        #create a trellis for each sentence, num_rows is number of unique tags, num_cols is words in current sentence
        trellis = {}
        sentence_length = len(sentence)
        last_col_idx = sentence_length - 1
        for row in range(v_tags): #row is tag index
            trellis[row] = {}
            for col in range(sentence_length): #col is word index in sentence, trellis[pos_tag_idx][word_index] = (prev_tag_idx, previous_word_idx, v (cost so far))
               trellis[row][col] = (row, col, 0) 
        for k, word in enumerate(sentence):
            for a, tagA in enumerate(tag_list):
                if k != 0:
                    emission_probability = getEmissionProbability(word_tag, word, tagA, v_words, total_tags, alpha)
                    possible_probs = {}
                    for b, tagB in enumerate(tag_list): #iterate through all possible tags B
                        transition_probability = getTransitionProbability(tagB, tagA, transition, v_tags, total_tags, beta)
                        v = trellis[b][k-1][2] #cost so far 
                        possible_probs[(b, k-1)] = v + transition_probability + emission_probability
                    bestNode = getMaxKey(possible_probs)
                    trellis[a][k] = (bestNode[0], bestNode[1], possible_probs[bestNode]) 
                else:
                    if tagA in init_tags:
                        probability = (init_tags[tagA] + gamma) / (v_init + gamma*(v_tags + 1))
                    else:
                        probability = (gamma) / (v_init + gamma*(v_tags + 1))
                    v = math.log(probability) + getEmissionProbability(word_tag, word, tagA, v_words, total_tags, alpha)
                    trellis[a][k] = (start, start, v)
        #done with building trellis, now go backwards starting from last col 
        tag_sequence = []
        curr_col = [trellis[row][last_col_idx] for row in trellis] #grab the last column
        curr_probs = [node[2] for node in curr_col]
        best_idx = curr_probs.index(max(curr_probs))
        tag_sequence.append(tag_list[best_idx])
        curr = (trellis[best_idx][last_col_idx][0], trellis[best_idx][last_col_idx][1])
        while curr != (start, start):
            tag_sequence.insert(0,tag_list[curr[0]])
            curr = (trellis[curr[0]][curr[1]][0], trellis[curr[0]][curr[1]][1])
        predictions.append(list(zip(sentence, tag_sequence)))
    return predictions

def getMaxKey(d):
    max_key = ""
    max_val = -999999999
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key 

def getEmissionProbability(word_tag, word, tag, v_words, total_tags, alpha):   
    word_tag_count = 0
    if (word,tag) in word_tag:
        word_tag_count = word_tag[(word,tag)]
    probability = (word_tag_count + alpha) / (total_tags[tag] + alpha*(v_words + 1))
    return math.log(probability)

def getTransitionProbability(prev_tag, curr_tag, transition, v_tags, total_tags, beta):
    tr_count = 0
    if (prev_tag, curr_tag) in transition:
        tr_count = transition[(prev_tag, curr_tag)]
    probability = (tr_count + beta) / (total_tags[prev_tag] + beta*(v_tags + 1))
    return math.log(probability)