import numpy as np

import random

from sklearn.metrics import accuracy_score, f1_score, precision_score
import time
import gzip
import _pickle as pkl
import codecs


#from _future import _pickle
import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Merge
from docutils.languages.af import labels
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential,model_from_json
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, GRU, Reshape, AveragePooling1D, RepeatVector

outputName="model1"
embeddingsDim = 50
batchSize = 128
hiddenSize = 50 #SAME AS Embeddings
#posSize = 100
epochs = 1000
maxLen = 0
maxLenSentence = 0
print("done till here 0")
f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

label2Idx = embeddings['label2Idx']
wordEmbeddings = np.asarray(embeddings['wordEmbeddings'])
#aspectEmbeddings = np.asarray(embeddings['aspectEmbeddings'])
print(wordEmbeddings[0].shape)
#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

f = gzip.open('pkl/data.pkl.gz', 'rb')
train_data = pkl.load(f)
dev_data = pkl.load(f)
test_data = pkl.load(f)
f.close()




trainingLeft=[]
trainingRight=[]
trainingAspects=[]
trainingLabels=[]
tuningLeft=[]
tuningRight=[]
tuningAspects=[]
tuningLabels=[]
testingLeft=[]
testingRight=[]
testingAspects=[]
testingLabels=[]
for i in train_data:
    trainingLeft.append(i[0])
    trainingRight.append(i[1])
    trainingLabels.append(i[3])
    trainingAspects.append(i[2])
for i in dev_data:
    tuningLeft.append(i[0])
    tuningRight.append(i[1])
    tuningLabels.append(i[3])
    tuningAspects.append(i[2])
for i in test_data:
    testingLeft.append(i[0])
    testingRight.append(i[1])
    testingLabels.append(i[3])
    testingAspects.append(i[2])

np.random.seed(1337) #reproduce
maxLeft=15   #for now
maxRight=15
maxAspect=15


label_size = 4 #len(labels)
#labels=label2Idx
train_x_left_pad = sequence.pad_sequences(trainingLeft,maxlen=maxLeft)
train_x_right_pad = sequence.pad_sequences(trainingRight,maxlen=maxRight)
train_x_np = sequence.pad_sequences(trainingAspects,maxlen=maxAspect)

tune_x_left_pad = sequence.pad_sequences(tuningLeft,maxlen=maxLeft)
tune_x_right_pad = sequence.pad_sequences(tuningRight,maxlen=maxRight)
tune_x_np = sequence.pad_sequences(tuningAspects,maxlen=maxAspect)

test_x_left_pad = sequence.pad_sequences(testingLeft,maxlen=maxLeft)
test_x_right_pad = sequence.pad_sequences(testingRight,maxlen=maxRight)
test_x_np = sequence.pad_sequences(testingAspects,maxlen=maxAspect)

leftInput = Input(shape=(maxLeft,),dtype='int32')
rightInput = Input(shape=(maxRight,),dtype='int32')
npInput = Input(shape=(maxAspect,),dtype='int32')

shared_embedding = Embedding(len(wordEmbeddings),embeddingsDim)


embLeft = shared_embedding(leftInput)
embRight = shared_embedding(rightInput)
embNP = shared_embedding(npInput)

npLSTMf = LSTM(hiddenSize)(embNP)
npLSTMf = Dropout(0.5)(npLSTMf)

embNPRepeatLeft = RepeatVector(maxLeft)(npLSTMf)
embNPRepeatRight = RepeatVector(maxRight)(npLSTMf)

embLeft = merge([embLeft,embNPRepeatLeft],mode='concat',concat_axis=-1)
embRight = merge([embRight,embNPRepeatRight],mode='concat',concat_axis=-1)

lstmLeftf = LSTM(hiddenSize)(embLeft)
lstmLeftf = Dropout(0.5)(lstmLeftf)


lstmRightf = LSTM(hiddenSize)(embRight)
lstmRightf = Dropout(0.5)(lstmRightf)

merged = merge([lstmLeftf,lstmRightf],mode='concat',concat_axis=-1)
output = Dense(label_size,activation='softmax')(merged)
model = Model(input = [leftInput,rightInput, npInput],output=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')

print("Model compilation done ")
model.summary()
#trainingLabels=np.zeros((2220,1))
labels=label2Idx
for e in range(epochs):
    print ("Epoch: " + str(e))
    
    model.fit([train_x_left_pad, train_x_right_pad, train_x_np], trainingLabels, batch_size=batchSize, epochs=1)

    # tuning
    predict = model.predict([tune_x_left_pad, tune_x_right_pad, tune_x_np],batch_size=1024)
    #this returns the predicted values for given batch in follwoing format
    #print(predict)
    y_pred=[]
    y_true=[]
    for i in range(len(predict)):
        predictIndex = np.argmax(predict[i])
        predictLabel = predictIndex #single classification
        #print(predictLabel)
        #print("\t")
        #print(tuningLabels[i])
        #print("\t")
        correctLabel = np.argmax(tuningLabels[i])
        #print(correctLabel)
        #print("end")
    

    #testing
    predict = model.predict([test_x_left_pad, test_x_right_pad, test_x_np],batch_size=1024)
    # write to file

    for i in range(len(predict)):
        predictIndex = np.argmax(predict[i])
        predictLabel = predictIndex
        correctLabel = np.argmax(testingLabels[i]) #converts [0 0 1 0] to 2
        y_pred.append(predictLabel)
        y_true.append(correctLabel)
    print("Model accuracy: "+str(accuracy_score(y_true,y_pred)))
    print("F1 score macro: "+str(f1_score(y_true,y_pred,average="macro")))
    y_pred=[]
    y_true=[]

   
#def compute_f1(pred)

#json_string = model.to_json()
#fjson = codecs.open('models/' + outputName + '.json','w',encoding='utf-8')
#fjson.write(json_string)
#fjson.close()
