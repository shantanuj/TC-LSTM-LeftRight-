import nltk 
import glob
import pprint
import re
import string
import sklearn
from multiprocessing import Pool
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sb
import itertools
import base64
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer
from hashlib import sha1
#from datasketch import MinHash


inputFiles=""
df_preProcess=None
fileType=""
fInputFile=""
delimiter=","
df_resultant=""
def process_XML(fileT):
        None
def listFiles(fileT):
    global inputFiles
    global fileType
    fileType=fileT
    inputFiles=sorted(glob.glob("data/*.{0}".format(fileType)))
    print("Found data files:")
    print(inputFiles)

def chooseFile(fileid,fileType):
    global fInputFile
    fInputFile=inputFiles[fileid-1]
    return fInputFile

def input_File_CSV(delim):
    global delimiter
    global df_preProcess
    delimiter=delim
    try:
        df_preProcess=None
        pdFile=pd.read_csv(fInputFile,encoding="ISO-8859-1",sep=delimiter,header=None)
        df_preProcess=pd.DataFrame(pdFile)
        print("The current read in file is as shown:\n")
        print("Incase the file is missing any values/columns, try changing the delimiter\n ")
    except:
        print("An error occured during reading file. Try changing the delimiter and run both cells again")
    finally:
        return df_preProcess

def processDataFrame(text_colid,label_colid,text_startRow,text_endRow):
    global df_resultant
    try:
        df_resultant=[]
        df_resultant=df_preProcess[[text_colid,label_colid]]
        df_resultant.columns=["Text","Label"]
        print("Formatted file:")
    
    except:
        print("Error. Try changing the column id (remember first column is id 0)")
    
    df_resultant=df_resultant.ix[text_startRow:text_endRow] 
    return df_resultant

def jsonDefault(object):
    return object.__dict__
#RUN AS IS
class Data_Operations_Pre_Training:
    #is it possible to add new functions to pandas file, so can call pdfile.data_Clean()
    #perhaps use inheritance
    #currently do as dops=Data_Operations(pdfile) then dops.data_Clean()
    #stuff to do 
    #1) Make functions more efficient--> 
    # 1.1 wordcorpus generation-> n^2 
    #1.1 sort list first nlogn, remove duplicates n 
    # make tokenization first
    # 2) try not using freqdist cause need to tokenize then merge lists
    # 3) Standardize usage of self.tokenizedList across functions
    
    def __init__(self,pfile): #constructor
        self.pfile=pfile
        preTextList=pfile['Text'].tolist()
        self.textList=[text.replace(u'\xa0',u' ') for text in preTextList   ] #replace space encoding of iso
        self.copyTextList=self.textList #mantain copy for reset
        preLabelList=pfile['Label'].tolist()
        self.labelList=[text.replace(u'\xa0',u' ') for text in preLabelList ]#replace space encoding of iso
        self.copyLabelList=self.labelList
        punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        self.fdist=None
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        self.wordCorpus={}
        pd.options.display.mpl_style = 'default'
        print("Object succesfully created")
        
#x=[i for i in list1]
#x
    def reset_changes(self):
        self.textList=self.copyTextList
        self.labelList=self.copyLabelList
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        self.updatePandas()
        return self.pfile
    def data_Clean(self):
         #tokens
         #punctuation tokens
        self.textToLower()
        self.dataRemovePunctuation()
        self.dataCleanStopWords()
    def addMissingValues(self): #fills empty columns
        None
    
    def mh_digest (data):
        m = MinHash(num_perm=512)
        for d in data:
            m.digest(sha1(d.encode('utf8')))
        return m

    def sortDatasetByLabel(self): #changes dataset to be structured by labels
        #sort a,b according to b zip or use numpy matrix
        #zippedList=zip(self.textList,self.labelList)
        #sortedZipped=sorted(list(zippedList),key=lambda x: x[1])
        #text,labels=zip(*sortedZipped)
        #self.textList=list(text)
        #self.labelList=list(labels)
        textVec=np.array(self.textList)
        labelVec=np.array(self.labelList)
        inds=labelVec.argsort()
        sortedTextVec=textVec[inds]
        self.textList=sortedTextVec.tolist()
        self.labelList=sorted(self.labelList)
        self.updatePandas()
        return self.pfile
    def textRaw(self):
        rawText=""
        for text in self.textList:
            rawText+=text+" "
        return rawText
    
    def makeWordCorpus(self,startIndex=None,endIndex=None):
        
        
        if(startIndex==None):
            startIndex=0
        if(endIndex==None):
            endIndex=len(self.textList)-1
        localTokenizedList=[word_tokenize(text) for text in self.textList[startIndex:endIndex+1]]
        all_tokens=[token for tokenL in localTokenizedList for token in tokenL if token not in stopwords.words("english")] #make more efficient
        fdist=nltk.FreqDist(all_tokens)
        self.fdist=fdist
        self.wordCorpus=dict(fdist)
        self.uniqueWordsSet=set(self.wordCorpus)
        return self.wordCorpus
        #word_freq
    """   
    def makeCustomWordCorpus(self,startIndex,endIndex,unique=True,wlength=0): #returns a customized word corpus 
        localWordCorpus=set()
        if(not unique):
            localWordCorpus={}
        localTokenizedList=[word_tokenize(text) for text in self.textList[startIndex,endIndex+1]]
        for text in self.tokenizedList:
            for token in text:
                tCount=0
                if (len(token)>wlength and not token in stopwords.words('english')): #fix this to incorporate smaller words
                    if(unique):
                        localWordCorpus.add(token)
                    else:
                        if(token in localWordCorpus.keys()): #in dictionary
                            tCount=localWordCorpus[token]
                            localWordCorpus[token]=tCount+1
                        else:
                            localWordCorpus[token]=tCount
        return localWordCorpus    
    """
    """
    def makeWordCorpus(self): #creates unique words corpus
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        self.data_Clean()
        for text in self.tokenizedList:
            for token in text:
                if len(token)>3 and not token in stopwords.words('english'): #fix this to incorporate smaller words
                    self.wordCorpus.add(token)
        self.textList=self.copyTextList
    """            
    def dataRemovePunctuation(self): #modifies panda dataframe without special characters/punctuation 
        noPuncTextList=[]
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        for tokenizedText in self.tokenizedList:
            new_text=[]
            for token in tokenizedText:
                new_token=regex.sub(u'',token)
                if not new_token == u'':
                    new_text.append(new_token)
            noPuncTextList.append(new_text)
        self.tokenizedList=noPuncTextList
        self.reconstructTextList()
        self.updatePandas()
        return self.pfile
        #p=Pool()#split process
        #for text in localTextList:
        #    print(list(itertools.chain.from_iterable(p.map(nltk.tokenize(text)))))
        #p.close()   
    
    def reconstructTextList(self): #object method to reconstruct text after tokenization
        temp_list=[]
        
        for text in self.tokenizedList:
            temp_list.append(' '.join(text))
            
        self.textList=temp_list
            
    def indexFile(self,startIndex,endIndex):
        self.textList=self.textList[startIndex:endIndex+1]
        self.labelList=self.labelList[startIndex:endIndex+1]
        self.updatePandas()
    def textToLower(self): #converts dataset to lowercase
        temp_list=[]
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        for text in self.textList:
            temp_list.append(text.lower())
        self.textList=temp_list
        self.updatePandas()
        return self.pfile
    def dataCleanStopWords(self): #modifies panda dataframe without stopwords
       
        self.tokenizedList=[word_tokenize(text) for text in self.textList]
        noStopWordsList=[]
        for text in self.tokenizedList:
            new_text_tokens=[]
            for word in text:
                if not word in stopwords.words('english'):  #make more efficient
                    new_text_tokens.append(word)
            noStopWordsList.append(new_text_tokens)
        self.tokenizedList=noStopWordsList
        self.reconstructTextList()
        self.updatePandas()
        return self.pfile
    def onlyAlphabets(self): #Removes numerals and non alphabets
        regex = re.compile('[^a-zA-Z]')
        
        onlyAlphabetsList=[]
        for tokenizedText in self.tokenizedList:
            new_text=[]
            for token in tokenizedText:
                new_token=regex.sub(u'',token)
                if not new_token == u'':  #DECIDE ON ' ' VS ''
                    new_text.append(new_token)
            onlyAlphabetsList.append(new_text)
        self.tokenizedList=onlyAlphabetsList
        self.reconstructTextList()
        self.updatePandas()
        return self.pfile
    def plot_freqdist_freq(self,max_num=None,cumulative=False,title='Frequency plot',linewidth=4):
        if(len(self.wordCorpus)==0):
            self.makeWordCorpus()
        tmp = self.fdist.copy()
        norm = self.fdist.N()
        for key in tmp.keys():
            tmp[key] = float(self.fdist[key]) / norm  #normalize count
        if max_num:
            tmp.plot(max_num, cumulative=cumulative,title=title, linewidth=linewidth)
        else:
            tmp.plot(cumulative=cumulative,title=title,linewidth=linewidth)

        return
        
    def plotFrequencies(self,maxnum=None):
        #if(len(self.wordCorpus)==0):
            #self.makeWordCorpus()
        self.plot_freqdist_freq(maxnum,False,'Frequency plot',2)

    def generalAnalysis(self):
        if(len(self.wordCorpus)==0):
            self.makeWordCorpus()
        
        print("Total number of words in text:")
        print(self.fdist.N())

        print("The 10 most occuring words with frequency of occurence")
        plotFrequencies(10)

        print("Distinct labels:")

    def tokenAnalyze(self,token): #returns stuff (probability, ) about a particular word
        if(len(self.wordCorpus)==0):
            self.makeWordCorpus()
        

        print("Occurences of word '%s' in text:"%(token))
        print(self.fdist[token])

        print("Frequency of word '%s' in text"%(token))
        print(self.fdist.freq(token))

        print("Most common cooccuring words with word '%s'"%(token))
        print("Not currently available")


    def dataPlot(self):        
        
        self.pfile.boxplot()

        self.pfile.hist() 
        self.pfile.groupby('Label').hist()
        self.pfile.groupby('Label').hist()
        self.data.groupby('Label').plas.hist(alpha=0.4)   
    #def fixWords(self):  #input text sometimes has words that need to be separated 
       # None   
    def labelAnalysis(self): #label analysis (top 5 words in label)
        None
    def labelCompare(self,label1,label2): #Use minhash for efficiency 
        label1_words=[]
        label2_words=[]
        m1set=mh_digest(set(label1_words))
        m2set=mh_digest(set(label2_words))
        estimjaccardSim=m1set.jaccard(m2set)

        s1 = set(label1_words)
        s2 = set(label2_words)

        actual_jaccard=float(len(s1.intersection(s2)))/float(len(s1.union(s2)))


    def mostCommonWords(self,num=None): #plot top 10 most common words
        #if(num)
        if(len(self.wordCorpus)==0):
            self.makeWordCorpus()
        print("Most common words plot")
        self.plotFrequencies(num)
        print("%s Most common words are: "%(num))
        print(self.fdist.most_common(num))
    def updatePandas(self): #returns modfied panda dataframe
        self.pfile= pd.DataFrame(
        {'Text':self.textList,
        'Label':self.labelList 
        })
    def viewPandas(self):
        print(self.pfile)
    def writeCSV(self): #writes the panda dataframe to csv
        #print(delimiter)
        return self.pfile.to_csv(sep=',') #default delimiter
    
        


#pdOpsObject.labelList
#pdOpsObject.data_Clean()
