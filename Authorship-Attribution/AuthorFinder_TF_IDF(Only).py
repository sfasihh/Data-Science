import annoy
from annoy import AnnoyIndex
import re
import os
import numpy as np
import pandas as pd
from pprint import pprint
import pickle

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.similarities.index import AnnoyIndexer
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
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

# Creating Corpus -----------------------------------------------------------------------------
path = 'C:\\Users\\samiy\\Desktop\\sam\\collection'

files = []
final = []
author_name = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
            author_name.append(file)

for f in files:
    temp=open(f, 'r')
    try:
        final.append(temp.read().strip())
        temp.close()
    except:
        temp.close()
        continue


DF = {}
with open('DF.pickle', 'rb') as f:
    DF = pickle.load(f)

#Function to determine tf_idf scores for individual queries
def TF(x):
    stop_words = stopwords.words('english')
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    x = x.split(' ')
    new_text = []
    for i in range(len(x)):
        if x[i] not in stop_words:
            temp = nlp(x[i])
            for i in temp:
                new_text.append(i.lemma_)
    x = new_text
    for i in symbols:
        x = np.char.replace(x, i, '')
        x = list(x)
    
    temp_tf = []
    for token in DF:
        counter = 0
        if token in x:
            for i in x:
                if token == i:
                    counter = counter +1
            temp_tf.append(counter/len(x))
        else:
            counter = 0
            temp_tf.append(0)
    
    return temp_tf


def Kprecision(path, author_name, title_choice):
    k=0
    relevant = []
    for f in author_name:
        a= path + "\\" + f
        temp= open(a, 'r')
        a= temp.read().split('.')
        for x in a:
            x= x+'.'
            x=x.strip()
            if x == title_choice:
                k=k+1
                relevant.append(f)
    
    return k, relevant


temp_file = datapath("model2")
lda = gensim.models.ldamodel.LdaModel.load(temp_file)
index = AnnoyIndex(len(DF))
index.load('TF_IDF.ann')

Mean = []

while 1:
    retrieved = []
    union = 0
    title_choice = input("Enter the research paper title: ")
    tmp = TF(title_choice)
    similarityList=index.get_nns_by_vector(tmp, 10)
    print(similarityList)
    for i in similarityList:
        print(author_name[i])
        retrieved.append(author_name[i])
    
    
    k, relevant = Kprecision(path, author_name, title_choice)
    for r in retrieved:
        for t in relevant:
            if r == t:
                union = union +1
    # Precision and Recall Calculation    
    print("Precision at " + str(10))
    print((union/len(retrieved))*100)
    print("Recall at " + str(len(relevant)))
    print((union/len(relevant))*100)
    # MAP Calculation 
    Mean.append(union/len(retrieved))
    print("Average Precision of Algorithm:")
    print(sum(Mean)/len(Mean))    
