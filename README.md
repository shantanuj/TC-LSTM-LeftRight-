# TC-LSTM 

Implementation of Duyu Tang's paper on using TC-LSTM for Aspect based sentiment analysis (https://arxiv.org/pdf/1512.01100.pdf)
Keras model with Tensorflow backend

### Components:
#### 1) Dataset: 
+ Semeval 14 dataset task 5- restaurant provided with labelling. 
+ Enron corpus dataset with labels (done by EEE@NTU)

#### 2) Preprocessing:
+ Two files __prepre.py and preprocessing.py__

+ Preprocessing initializes word vectors and stores in pickle file.
+ Include preprocessing of xml data source for semeval and dataset cleaning model. (TO DO)
+ Generalize for different datasource types.
+ 
+ prepre.py is a convenient way to format and perform rudimentary analysis of the raw text datasource; and convert to compatible csv format for preprocessing.py
+ It only supports csv files as of now 
+ To do: Include jupyter notebook interface and XML reading for semeval.


#### 3) Model:
+ Implemented in Keras with **accuracy of 74% with Semeval restaurant terms with 300d vectors on Nvidia GPU.**
+ Second model of Bidirectional, target LSTM and single lstm included (TO DO)



### How to run:
1) Download word embeddings of choice. Recommended glove (https://nlp.stanford.edu/projects/glove/) 300d dataset for optimum performance and store in embeddings directory.
2) Select data file in data directory by setting path in preprocessing.py
3) Run preprocessing.py: python3 preprocessing.py to generate word vectors and embeddings.
4) Change embeddings path and embeddingsdim and hiddendim to chosen embeddings size in model.py
5) Run model.py: python3 model.py (Should automatically perform GPU initialization if workstation used)

### To Do:
1) Incorporate python command line arguments to choose embeddings and data.
2) Improve aspect classification in preprocessing.
3) Generalize prepre and preprocessing dependency to support different text data sources
4) Make code py2 compatible to resolve NVidia workstation problem.
