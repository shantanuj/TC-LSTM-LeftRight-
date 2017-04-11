# Bilstm2

Implementation of Duyu Tang's paper on using TC-LSTM for Aspect based sentiment analysis (https://arxiv.org/pdf/1512.01100.pdf)
Keras model

Components:
1) Dataset: ]
a) Semeval 14 dataset task 5- restraunt provided with labelling. 
b) Enron corpus dataset with labels (done by EEE@NTU)

2) Preprocessing:
a) Preprocessing initializes word vectors and stores in pickle file.
b) Include preprocessing of xml data source for semeval and dataset cleaning model. (TO DO)

3) Model:
a) Implemented in Keras with accuracy of 74% with Semeval rest terms with 300d vectors.
b) Second model of Bidirectional and single lstm included (TO DO)



How to run:
1) Download word embeddings of choice. Recommended glove (https://nlp.stanford.edu/projects/glove/) 300d dataset for optimum performance and store in embeddings directory.
2) Select data file in data directory.
3) 

To Do:
1) Incorporate python command line arguments to choose embeddings and data.
2) Improve aspect classification in preprocessing.
