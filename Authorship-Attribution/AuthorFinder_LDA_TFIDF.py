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

#Function to retrieve corpus/topic distributions of individual queries. 
def getCorpus(final):
    # Convert to list
    data = [final]
    print(data)
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
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    #print(data_lemmatized[:1])
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    dist = []
    
    for l in corpus:
        dist.append(lda[l])
    counter=-1    
    for i in dist:
        counter=counter+1
        test=i[0]
        v = []
        for l in range(20):
            v.append(0)
        for l in test:   
            v[l[0]]=l[1]
    print(v)
    return corpus, v

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
index.load('similarityTest8.ann')

Mean = []

while 1:
    retrieved = []
    union = 0
    title_choice = input("Enter the research paper title: ")
    tmp = TF(title_choice)
    corpus, vector = getCorpus(title_choice)
    finalV = vector + tmp
    similarityList=index.get_nns_by_vector(finalV, 10)
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
