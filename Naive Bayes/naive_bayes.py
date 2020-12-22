# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
    
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
                 'there', 'about', 'once', 'during', 'out', 'very', 'having', 
                 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 
                 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 
                 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 
                 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 
                 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
                 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',
                 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 
                 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 
                 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 
                 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 
                 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 
                 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 
                 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    neg_prior = 1 - pos_prior
    predicted_labels = []
    bag_of_words_pos = {}
    bag_of_words_neg = {}
    pos_probs = {}
    neg_probs = {}
    n_pos = 0 
    n_neg = 0
    v_pos = 0
    v_neg = 0
    # build dictionaries of positive and negative words and their counts 
    for i, review in enumerate(train_set):
        for word in review:
            if word not in stopwords:
                if train_labels[i] == 1:
                    n_pos += 1
                    if word not in bag_of_words_pos:
                        bag_of_words_pos[word] = 1
                        v_pos += 1
                    else:
                        bag_of_words_pos[word] += 1
                else:
                    n_neg += 1
                    if word not in bag_of_words_neg:
                        bag_of_words_neg[word] = 1
                        v_neg += 1
                    else:
                        bag_of_words_neg[word] += 1       
    for word in bag_of_words_pos:
        pos_probs[word] = (bag_of_words_pos[word] + smoothing_parameter)/(n_pos + (smoothing_parameter*(v_pos + 1)))
    for word in bag_of_words_neg:
        neg_probs[word] = (bag_of_words_neg[word] + smoothing_parameter)/(n_neg + (smoothing_parameter*(v_neg + 1)))

    for review in dev_set:
        pos = 0
        neg = 0
        for word in review:
            if word not in stopwords:
                if word not in pos_probs:
                    pos_probs[word] = (smoothing_parameter)/(n_pos + (smoothing_parameter*(v_pos + 1)))
                if word not in neg_probs:
                    neg_probs[word] = (smoothing_parameter)/(n_neg + (smoothing_parameter*(v_neg + 1)))
                pos += math.log(pos_probs[word])
                neg += math.log(neg_probs[word])
        pos += math.log(pos_prior)
        neg += math.log(neg_prior)
        label = 0 if neg > pos else 1
        predicted_labels.append(label)
    return predicted_labels

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.0325, bigram_smoothing_parameter=0.0625, bigram_lambda=0.05, pos_prior=0.8):
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
                 'there', 'about', 'once', 'during', 'out', 'very', 'having', 
                 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 
                 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 
                 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 
                 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 
                 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
                 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',
                 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 
                 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 
                 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 
                 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 
                 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 
                 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 
                 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    neg_prior = 1 - pos_prior
    predicted_labels = []
    bag_of_words_pos = {}
    bag_of_words_neg = {}
    pos_probs = {}
    neg_probs = {}
    n_pos = 0 
    n_neg = 0
    v_pos = 0
    v_neg = 0
    for i, review in enumerate(train_set):
        for word in review:
            if word not in stopwords:
                if train_labels[i] == 1:
                    n_pos += 1
                    if word not in bag_of_words_pos:
                        bag_of_words_pos[word] = 1
                        v_pos += 1
                    else:
                        bag_of_words_pos[word] += 1
                else:
                    n_neg += 1
                    if word not in bag_of_words_neg:
                        bag_of_words_neg[word] = 1
                        v_neg += 1
                    else:
                        bag_of_words_neg[word] += 1       
    for word in bag_of_words_pos:
        pos_probs[word] = (bag_of_words_pos[word] + unigram_smoothing_parameter)/(n_pos + (unigram_smoothing_parameter*(v_pos + 1)))
    for word in bag_of_words_neg:
        neg_probs[word] = (bag_of_words_neg[word] + unigram_smoothing_parameter)/(n_neg + (unigram_smoothing_parameter*(v_neg + 1)))
# ----------------------------------------------------- Bigram Calculation----------------------------------------------------#    
    bag_of_phrases_pos = {}
    bag_of_phrases_neg = {}
    pos_phrase_probs = {}
    neg_phrase_probs = {}
    n_phrase_pos = 0 
    n_phrase_neg = 0
    v_phrase_pos = 0
    v_phrase_neg = 0
    for i, review in enumerate(train_set):
        for j, word in enumerate(review):
            if j == len(review) - 1:
                break 
            if word not in stopwords and review[j + 1] not in stopwords:
                phrase = (word, review[j + 1])
                phrase_mirror = (review[j + 1], word)
                if train_labels[i] == 1:
                    n_phrase_pos += 1
                    if phrase not in bag_of_phrases_pos or phrase_mirror not in bag_of_phrases_pos:
                        bag_of_phrases_pos[phrase] = 1
                        v_phrase_pos += 1
                    else:
                        bag_of_phrases_pos[phrase] += 1
                else:
                    n_phrase_neg += 1
                    if phrase not in bag_of_phrases_neg or phrase_mirror not in bag_of_phrases_neg:
                        bag_of_phrases_neg[phrase] = 1
                        v_phrase_neg += 1
                    else:
                        bag_of_phrases_neg[phrase] += 1   
    for phrase in bag_of_phrases_pos:
        pos_phrase_probs[phrase] = (bag_of_phrases_pos[phrase] + bigram_smoothing_parameter)/(n_phrase_pos + (bigram_smoothing_parameter*(v_phrase_pos + 1)))
    for phrase in bag_of_phrases_neg:
        neg_phrase_probs[phrase] = (bag_of_phrases_neg[phrase] + bigram_smoothing_parameter)/(n_phrase_neg + (bigram_smoothing_parameter*(v_phrase_neg + 1)))    
#------------------------------------------------Predict on development data set-----------------------------------------------#
    for review in dev_set:
        pos = 0
        neg = 0
        pos_uni = 0
        neg_uni = 0
        for word in review:
            if word not in stopwords:
                if word not in pos_probs:
                    pos_probs[word] = (unigram_smoothing_parameter)/(n_pos + (unigram_smoothing_parameter*(v_pos + 1)))
                if word not in neg_probs:
                    neg_probs[word] = (unigram_smoothing_parameter)/(n_neg + (unigram_smoothing_parameter*(v_neg + 1)))
                pos_uni += math.log(pos_probs[word])
                neg_uni += math.log(neg_probs[word])
        pos_uni += math.log(pos_prior)
        neg_uni += math.log(neg_prior)         
        pos_bi = 0
        neg_bi = 0
        for i, word in enumerate(review):
            if i == len(review) - 1:
                break 
            if word not in stopwords and review[i + 1] not in stopwords:
                phrase = (word, review[i + 1])
                phrase_mirror = (review[i + 1], word)
                if phrase not in pos_phrase_probs or phrase_mirror not in pos_phrase_probs:
                    pos_phrase_probs[phrase] = (bigram_smoothing_parameter)/(n_phrase_pos + (bigram_smoothing_parameter*(v_phrase_pos + 1)))
                if phrase not in neg_phrase_probs or phrase_mirror not in neg_phrase_probs:
                    neg_phrase_probs[phrase] = (bigram_smoothing_parameter)/(n_phrase_neg + (bigram_smoothing_parameter*(v_phrase_neg + 1)))
                pos_bi += math.log(pos_phrase_probs[phrase])
                neg_bi += math.log(neg_phrase_probs[phrase])
        pos_bi += math.log(pos_prior)
        neg_bi += math.log(neg_prior)
        pos = (1 - bigram_lambda)*pos_uni + (bigram_lambda*pos_bi)
        neg = (1 - bigram_lambda)*neg_uni + (bigram_lambda*neg_bi)
        label = 0 if neg > pos else 1
        predicted_labels.append(label)
    return predicted_labels