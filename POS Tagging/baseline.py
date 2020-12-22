"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
'''
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def baseline(train, test):
    words = {} #nested dicionary {word1: tags: {}, word2: tags: {}...}
    tags = {}
    predicted =[]
    for sentence in train:
        for word, tag in sentence:
            if word not in words: 
                words[word] = {}
                words[word][tag] = 1
            else:
                if tag not in words[word].keys():
                    words[word][tag] = 1
                else:
                    words[word][tag] += 1 
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1
    best_tag = findMax(tags)
    for sentence in test:
        temp = []
        for word in sentence: 
            if word not in words:
                temp.append((word, best_tag))
            else:
                temp.append((word, findMax(words[word])))
        predicted.append(temp)
    return predicted

def findMax(d):
    curr_max = -1
    best = ""
    for key, val in d.items():
        if val > curr_max:
            curr_max = val 
            best = key 
    return best 
