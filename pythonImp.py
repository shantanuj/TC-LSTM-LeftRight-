import csv 
import numpy as np 
from unicode import unicode
import theano
import gzip
import cPickle as pkl

vocabPath='/embeddings/glove.6B.50d.txt'  #use the OPTIMIZED ONE LATER

picklePath='pkl/data.pkl.gz'
embeddingsPicklePath='pkl/embeddings.pkl.gz'

exampleFile=open('RestAspTermABSA.csv')
reader=csv.reader(exampleFile,delimiter=';')
edata=list(reader)
npedata=np.asarray(edata)
text_pre=npedata[:,0].tolist() #sentences 3699

windowSize=3


aspect_pre=npedata[:,1:2].tolist() #target words
i=0
while(i<len(aspect_pre)):
	aspect_pre[i]=aspect_pre[i][0]
	i+=1
label_pre=npedata[:,2:3].tolist()
i=0
while(i<len(label_pre)):
	label_pre[i]=label_pre[i][0]
	i+=1
i=0
while(i<len(label_pre)):
	if(label_pre[i]=='negative'):
		label_pre[i]=0
	elif(label_pre[i]=='neutral'):
		label_pre[i]=1
	else:
		label_pre[i]=2
	i=i+1

i=0
#labels now mapped and in a list



#now do word embeddings
#label2IdX={}
#map labls

#first extract relevant words that need embeddings (saves time)
"""
words=set()
for sentence in text_pre:
	sentence=sentence.strip()
	if(len(sentence)==0):
		continue
	splits=sentence.split('\t')
	words.add(splits[1])

words=sorted(words)  #T
"""



#Now the embeddings part

word2Idx={}
wordEmbeddings=[] #embeddings matrix

idx=0 #increment id 

for line in open(vocabPath,'r'):  
	split=line.strip().split(' ')
	word=split[0]
	if(len(word2Idx)==0):
		word2Idx["PADDING_TOKEN"]=len(word2Idx)
		vector=np.zeros(len(split)-1)
		wordEmbeddings.apend(vector)

		word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)




    else:
		wordEmbeddings.append(np.array([float(num) for num in split[1:]]))
		word2Idx[split[0]]=idx
	


	idx+=1

label2Idx={"negative":0,"neutral":1,"positive":2}

#store aspect embeddings here (LSTM-TC)
for word in aspect_pre:
	if(len)







fWordEmbeddings=np.asarray(wordEmbeddings,dtype='float32')

embeddings_size=fWordEmbeddings.shape[1] #in 50d it is 50

embeddings={'wordEmbeddings':wordEmbeddings,
'word2Idx':word2Idx,'label2Idx':label2Idx}


f=gzip.open(embeddingsPicklePath,'wb')
pkl.dump(embeddings,f,-1)
f.close()



##CREATING MATRICES

training_set=text_pre[:3400]
testing_set=text_pre[3400:]

train_set=createMatrices(training_set,windowSize,word2Idx,label2Idx,aspect_pre)
test_set=createMatrices(testing_set,windowSize,word2Idx,label2Idx,aspect_pre)


f=gzip.open(picklePath,'wb')
pkl.dump(train_set,f,-1)
pkl.dump(test_set,f,-1)
f.close()

print("Data stored in pkl folder")


#label2Idx={"negative":0}







def createMatrices(sentences,windowSize,word2Idx,label2IdX,aspect_pre):
	#label2Idx basicla
	unknownIdx=word2Idx['UKNOWN_TOKEN']
	paddingIdx=word2Idx['PADDING_TOKEN']
	dataset=[]
	wordcount=0
	unknowWordcount=0
	for sentence in sentences:
		wordIndices=[]
		aspectIndices=[]
		labelIndices=[]

		for word in sentence:
			wordCount+=1
			if word in word2Idx:
				wordIdx=word2Idx[word]
			elif word.lower() in word2Idx:
				wordIdx=word2Idx[word.lower()]
			else:
				wordIdx=unknownIdx
				unknowWordcount+=1

				


