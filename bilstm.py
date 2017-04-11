#This was based on Niel Reimer's code available in his course https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master

import numpy as np

import random


import time
import gzip
import cPickle as pkl


#import BIOF1Validation

import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from docutils.languages.af import labels




f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']
aspectEmbeddings = embeddings['aspectEmbeddings']

#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_data = pkl.load(f)
dev_data = pkl.load(f)
test_data = pkl.load(f)
f.close()




n_out = len(label2Idx)
tokens = Sequential()
tokens.add(Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False))

casing = Sequential()
casing.add(Embedding(output_dim=aspectEmbeddings.shape[1], input_dim=aspectEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)) 
  
model = Sequential();
model.add(Merge([tokens, casing], mode='concat'))  
model.add(Bidirectional(LSTM(10, return_sequences=True, dropout_W=0.2))) 
model.add(TimeDistributed(Dense(n_out, activation='softmax')))

sgd = SGD(lr=0.1, decay=1e-7, momentum=0.0, nesterov=False, clipvalue=3) 
rmsprop = RMSprop(clipvalue=3) 
model.compile(loss='adam', optimizer=sgd)

model.summary()



print "%d train sentences" % len(train_data)
print "%d dev sentences" % len(dev_data)
print "%d test sentences" % len(test_data)

for epoch in xrange(number_of_epochs):    
    print "--------- Epoch %d -----------" % epoch
    random.shuffle(train_data)
    for startIdx in xrange(0, len(train_data), stepsize):
        start_time = time.time()    
        for batch in iterate_minibatches(train_data, startIdx, startIdx+stepsize):
            labels, tokens, casing = batch       
            model.train_on_batch([tokens, casing], labels)   
        print "%.2f sec for training" % (time.time() - start_time)
        
        
        #Train Dataset       
        start_time = time.time()    
        predLabels, correctLabels = tag_dataset(train_data)        
        pre_train, rec_train, f1_train = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_train, rec_train, f1_train)
        
        #Dev Dataset       
        predLabels, correctLabels = tag_dataset(dev_data)        
        pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev)
        
        #Test Dataset       
        predLabels, correctLabels = tag_dataset(test_data)        
        pre_test, rec_test, f1_test= BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label)
        print "Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test)
        
        print "%.2f sec for evaluation" % (time.time() - start_time)
        print ""
 
 
 """
Computes the F1 score on tagged data
@author: Nils Reimers
"""


#Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label): 
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
            
    
    #print label_pred
    #print label_correct
    
    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in xrange(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': #A new chunk starts
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision
