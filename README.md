# TC-LSTM 

Implementation of Duyu Tang's paper on using TC-LSTM for Aspect based sentiment analysis (https://arxiv.org/pdf/1512.01100.pdf)
Keras model with Tensorflow backend

Components:
1) Dataset: 
a) Semeval 14 dataset task 5- restraunt provided with labelling. 
b) Enron corpus dataset with labels (done by EEE@NTU)

2) Preprocessing:
a) Preprocessing initializes word vectors and stores in pickle file.
b) Include preprocessing of xml data source for semeval and dataset cleaning model. (TO DO)

3) Model:
a) Implemented in Keras with accuracy of 74% with Semeval rest terms with 300d vectors on Nvidia GPU.
b) Second model of Bidirectional, target LSTM and single lstm included (TO DO)



How to run:
1) Download word embeddings of choice. Recommended glove (https://nlp.stanford.edu/projects/glove/) 300d dataset for optimum performance and store in embeddings directory.
2) Select data file in data directory.
3) Change embeddings path and embeddingsdim and hiddendim to chosen embeddings size in model.py
4) Run model.py: python3 model.py (Should automatically perform GPU initialization if workstation used)

To Do:
1) Incorporate python command line arguments to choose embeddings and data.
2) Improve aspect classification in preprocessing.
3) Make code py2 to resolve NVidia workstation problem.
