"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""
import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    W = np.zeros(train_set.shape[1])
    b = 0
    for epoch in range(max_iter):
        for i, image in enumerate(train_set):
            y_hat = np.sign(np.dot(W, image) + b)
            y = -1 if train_labels[i] != 1 else 1
            if y != y_hat:
                W += learning_rate * y * image 
                b += learning_rate * y
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    predictions = []
    for image in dev_set:
        out = np.dot(W, image) + b
        label = 1 if out > 0 else 0
        predictions.append(label)    
    return predictions

def classifyKNN(train_set, train_labels, dev_set, k):
    predictions = []
    for test_image in dev_set: 
        k_NN = []
        for i, train_image in enumerate(train_set):
            distance = np.linalg.norm(test_image - train_image)
            k_NN.append((i, distance))
        k_NN.sort(key=lambda pair:pair[1])
        k_NN = k_NN[:k]
        classes = []
        for pair in k_NN:
            label = 1 if train_labels[pair[0]] else -1
            classes.append(label)
        label = 1 if sum(classes) > 0 else 0
        predictions.append(label)
    return predictions