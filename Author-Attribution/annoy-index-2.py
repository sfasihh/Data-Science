import annoy
from annoy import AnnoyIndex
import re
import os
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.similarities.index import AnnoyIndexer
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

path = 'C:\\Users\\samiy\\Desktop\\sam\\collection'
path1 = 'C:\\Users\\samiy\\Desktop\\sam\\500papers'
files = []
author_name = []

# r=root, d=directories, f = files

for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            author_name.append(file)
			
def getCorpus(final):

    # Convert to list
    data = [final]
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
            
        
    def sent_to_words(sentences):

        for sentence in sentences:

            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)     

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)    

    # Define functions for stopwords, bigrams, trigrams and lemmatization

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    model = []   
    for c in corpus:
        model.append(lda[c])
    counter=-1    

    for i in model:
        counter=counter+1
        test=i[0]
        v = []

        for c in range(20):
            v.append(0)

        for c in test:   
            v[c[0]]=c[1]    
			
    return corpus, v
	
def precisionOfK(path1, author_name, title_paper): # number of relevant topics and number of retrieved documents / # of retrieved topics
	k = 0
	for a in author_name:
		test1 = open(path1+'\\'+a, "r")
		test2 = test1.read().split('.') 
		for y in test2:
			if y==title_paper: 
				k = k+1 
				retrieved = 10
	#print(k) # Application of Gene-Expression Programming in Hydraulic Engineering. // Asynchronous replica exchange software for grid and heterogeneous computing. // Process Algebra and Model Checking
	return k/10
	
# meanAvgP: sum of precisions divided queries. // place the summation of precision in array and divide by amount of times you've gone through precision loop 
#	m = 0
#	q = 500 # queries: amount of times you search for a document 
#	while (1 < m <= n):
#		map = precisionOfK(m)/n
#	print(map)
#	return map9
t = AnnoyIndex(20, 'angular')
path = 'C:\\Users\\samiy\\Desktop\\sam\\collection'
path1 = 'C:\\Users\\samiy\\Desktop\\sam\\500papers'
t.load('annoyIndex.ann')
map = []

while True:
	paper_title = input("Title of Paper: ")
	corpus, vector = getCorpus(paper_title)
	precision = precisionOfK(path, author_name, paper_title)
	similarity = t.get_nns_by_vector(vector, 10)
	# by item: authors, by vector: topic dist, annoy: # & topic dist
	
	for i in similarity:
		author = author_name[i]
		pvec = t.get_item_vector(i)
		print(author)
		print(pvec)
	# Metrics	
	print("precision of documents: ")
	print(precision)
	map.append(precision)
	print("Mean Average Precision (MAP):")
	print((sum(map)/len(map)))
		
#while True:
#	paper_title = input("Title of Paper: ")  # Asks user to input a paper title
#	corpus, vector = getCorpus(paper_title)	 # vector is a split array that stores probabilities of given input at topic indexes
#	print ("vector: ")
#	print(vector) 
#	print("corpus: ")
#	print(lda[corpus[0]]) # prints out topic distributions of the title the user enters
#	similarity = t.get_nns_by_vector(vector, 10) # finds 10 nearest authors by comparing already built model with vector distributions of title

#	for i in similarity:
#		author = author_name[i]
#		pvec = t.get_item_vector(i) # get_item_vector(i) will return the vector of previously added item i
#		print(author)
#		print(pvec)
	
	# If user inputs in title: Software. We get the output (only show an example of the first author):
	# Title of Paper: Software
	# vector:
	# [0.092502, 0, 0.030445527, 0.020234434, 0.05216685, 0.036428735, 0.019886222, 0.032177642, 0.057555523, 0.055731032, 0.031077918, 
	#  0.014560424, 0.039349094, 0.24711397, 0.013663915, 0.06507133, 0.0853662, 0.030136764, 0.03961218, 0.034620933]
	# corpus:
	# ([(0, 0.092502), (2, 0.030445527), (3, 0.020234434), (4, 0.05216685), (5, 0.036428735), (6, 0.019886222), (7, 0.032177642), (8, 0.057555523), 
	# (9, 0.055731032), (10, 0.031077918), (11, 0.014560424), (12, 0.039349094), (13, 0.24711394), (14, 0.013663915), (15, 0.06507133), 
	# (16, 0.0853662), (17, 0.030136764), (18, 0.03961218), (19, 0.034620933)], [(0, [13])], [(0, [(13, 0.9999927)])])
	# Clarisse Dhaenens.txt
	# [0.09250213205814362, 0.0, 0.030445566400885582, 0.02023446001112461, 0.05216692015528679, 0.0364287793636322, 0.01988624595105648, 
	# 0.03217768669128418, 0.057555604726076126, 0.05573111027479172, 0.031077958643436432, 0.014560442417860031, 0.03934914246201515, 
	# 0.24711297452449799, 0.01366393268108368, 0.06507141888141632, 0.08536633104085922, 0.030136801302433014, 0.039612237364053726, 
	# 0.0346209779381752]
	
	# As we can see, vector only prints out a split array of just topic distributions of "Software". Corpus is the annoy database which prints out 
	# an array of the index and topic distribution of "Software". The likeliest author to have written "Software" is Clairesse Dhaenens. Her topic
	# distribution for the first topic is very similar to the topic distribution of "Software", and for every index as we go on. 
	# This makes her the likeliest to have written that title paper.	