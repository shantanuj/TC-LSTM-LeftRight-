import csv 
import numpy as np 
#from unicode import unicode
import theano
import gzip
import _pickle as pkl
import string
import  re
vocabPath='embeddings/glove.6B.50d.txt'  #use the OPTIMIZED ONE LATER

picklePath='pkl/data.pkl.gz'
embeddingsPicklePath='pkl/embeddings.pkl.gz'

exampleFile=open('RestAspTermABSA.csv') #generalize this 
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

#MIGHT NOT NEED THIS PART
a="""
while(i<len(label_pre)):
	if(label_pre[i]=='negative'):
		label_pre[i]=0
	elif(label_pre[i]=='neutral'):
		label_pre[i]=1
	elif(label_pre[i]=='conflict'):
		label_pre[i]=2
	else:
		label_pre[i]=3
	i=i+1
print(label_pre)
"""
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
	wordEmbeddings.append(np.array([float(num) for num in split[1:]]))
	word2Idx[split[0]]=idx
	idx+=1
word2Idx["PADDING_TOKEN"]=len(word2Idx)
vector=np.zeros(len(split)-1)
wordEmbeddings.append(vector)

word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
vector=np.random.uniform(-0.25,0.25,len(split)-1)
wordEmbeddings.append(vector)
label2Idx={'negative':0,'neutral':1,"positive":2,"conflict":3}

#store aspect embeddings here (LSTM-TC)
#for word in aspect_pre:
	#if(len)







fWordEmbeddings=np.asarray(wordEmbeddings,dtype='float32')

embeddings_size=fWordEmbeddings.shape[1] #in 50d it is 50

embeddings={'wordEmbeddings':wordEmbeddings,
'word2Idx':word2Idx,'label2Idx':label2Idx}


f=gzip.open(embeddingsPicklePath,'wb')
pkl.dump(embeddings,f,-1)
f.close()


def SegmentLeftRight(sentences,aspect,labels,label2Idx):
	#returns [[[left words list],[right words list],[aspect words list],[label]i]
	segmentList=[]
	j=0
	
	#print(labels)
	while(j<len(sentences)):
		preAspectList=aspect[j].split(' ')
		preAllWords=sentences[j].split(' ')
		regex = re.compile('[%s]' % re.escape(string.punctuation))
		allWords=[]
		aspectList=[]
		for i in preAspectList:
			new_token=regex.sub(u'',i)
			if(not new_token == u''):
				aspectList.append(new_token)
		for i in preAllWords:
			new_token=regex.sub(u'',i)
			if(not new_token == u''):
				allWords.append(new_token)
		aspectWordIndex=int(len(allWords)/2)			#HAVE TO FIX
		if((aspectList[0] in allWords)):
			aspectWordIndex=allWords.index(aspectList[0])
		elif (aspectList[0]+"s" in allWords):
			aspectWordIndex=allWords.index(aspectList[0]+"s")
		leftWords=createIDVectors(allWords[:aspectWordIndex])
		rightWords=createIDVectors(allWords[aspectWordIndex+1:])#allWords.index(aspectList[len(aspectList)-1]):])

		aspectWords=createIDVectors(aspectList[0]) #CHANGE LATER CAUSE SHOULD BE MEAN TAKEN
		labelId=label2Idx[labels[j]]
		labels_list=[0,0,0,0]
		labels_list[labelId]=1
		segmentList.append([leftWords,rightWords,aspectWords,labels_list])#,label2Idx[labels[i]]])
		j+=1
	return segmentList


def createIDVectors(inputWords):
	unknownIdx=word2Idx['UNKNOWN_TOKEN']
	paddingIdx=word2Idx['PADDING_TOKEN']
	idvector=[]
	unknowWordcount=0
	wordCount=0
	for word in inputWords:
		wordCount+=1
		if(word in word2Idx):
   			wordIdx=word2Idx[word]
		elif word.lower() in word2Idx:
			wordIdx=word2Idx[word.lower()]
		else:
			wordIdx=unknownIdx
			unknowWordcount+=1
		idvector.append(wordIdx)
	return idvector	

def createMatrices(sentences,windowsize,word2Idx,label2IdX,aspect_pre,label_pre):
	"""#format
	#sentences--> ["Example is here ","this is great word",....]
	windowsize --> if implementation requires segmented words input
	word2Idx --> id of word in embeddings vector, only need id as KERAS already implements embedding layer
	label2Idx --> id of label, so negative basically 0 
	will return an array with [[word1id,awordid]
	"""
	unknownIdx=word2Idx['UNKNOWN_TOKEN']
	paddingIdx=word2Idx['PADDING_TOKEN']
	dataset=[]
	wordcount=0
	unknowWordcount=0
	labelIndices=[]
	i=0
	for sentence in sentences:
		
		wordIndices=[]
		aspectIndices=[]
		labelIndices=[0,0,0,0] #[negative neutral conflict positive]
		labelIndices[label2Idx[label_pre[i]]]=1
		#print(labelIndices)
		for word in sentence:
			wordcount+=1
			aspect_word=aspect_pre[i]
			if(aspect_word in word2Idx):
				aspectIndices.append(word2Idx[aspect_word])
			elif aspect_word.lower() in word2Idx:
				aspectIndices.append(word2Idx[aspect_word.lower()])
			else:
				aspectIdx=unknownIdx
				aspectIndices.append(aspectIdx)

			aspect_word_list=aspect_word.split()
			aspect_word=(aspect_word_list[0]).lower()
			#PROBLEM HERE when aspect is 2 words and USE lower case
			
			#PR1 aspect coupled with each word???
			#Alt aspect coupled only once along with word vector id so [word,aspect] instead of [word1id,2id],[aspect,aspect]
			if(word in word2Idx):
				wordIdx=word2Idx[word]
			elif word.lower() in word2Idx:
				wordIdx=word2Idx[word.lower()]
			else:
				wordIdx=unknownIdx
				unknowWordcount+=1

				
			wordIndices.append(wordIdx)
		i+=1

		dataset.append([wordIndices,aspectIndices,labelIndices])
	
	return dataset

##CREATING MATRICES

training_set=text_pre[:2220] #to pass to creatematrix in following format
#[sentence 1, sentece2 sentence 3]

training_set_segmented=SegmentLeftRight(training_set,aspect_pre[:2220],label_pre[:2220],label2Idx)


deving_set=text_pre[2220:3141]
deving_set_segmented=SegmentLeftRight(deving_set,aspect_pre[2220:3141],label_pre[2220:3141],label2Idx)
testing_set=text_pre[3141:]#3700
testing_set_segmented=SegmentLeftRight(testing_set,aspect_pre[3141:],label_pre[3141:],label2Idx)

print(testing_set_segmented[0])

#print(aspect_pre)




#train_set=createMatrices(training_set,windowSize,word2Idx,label2Idx,aspect_pre,label_pre)
#dev_set=createMatrices(deving_set,windowSize,word2Idx,label2Idx,aspect_pre,label_pre)
#test_set=createMatrices(testing_set,windowSize,word2Idx,label2Idx,aspect_pre,label_pre)


f=gzip.open(picklePath,'wb')
pkl.dump(training_set_segmented,f,-1)
pkl.dump(deving_set_segmented,f,-1)
pkl.dump(testing_set_segmented,f,-1)
f.close()

print("Data stored in pkl folder")


#label2Idx={"negative":0}







