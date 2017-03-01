import csv 
import numpy as np 

exampleFile=open('RestAspTermABSA.csv')
reader=csv.reader(exampleFile,delimiter=';')
edata=list(reader)
npedata=np.asarray(edata)
text_pre=npedata[:,0].tolist() #sentences 3699

training_set=text_pre[:3400]
testing_set=text_pre[3400:]


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

