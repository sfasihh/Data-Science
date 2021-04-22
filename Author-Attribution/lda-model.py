import re
import numpy as np
import pandas as pd
from pprint import pprint
import re
import os

#Annoy
import annoy
from annoy import AnnoyIndex

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.similarities.index import AnnoyIndexer
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import datapath
from gensim.test.utils import common_corpus
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
# Enable loggingfor gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import glob

all_files = glob.glob("/Users/samiya/Desktop/collection/*")
li=[]

df = pd.concat((pd.read_csv(f, sep="\n", names=["title"]) for f in all_files))

# Convert to list
data = df.title.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]


def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

files = []
final = []
author_name = []
path = 'C:\\Users\\samiy\\Desktop\\sam\\collection'

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file)) 
            author_name.append(file)

for f in files:
    temp = open(f, 'r', encoding = "utf8")
    final.append(temp.read())

data_words = list(sent_to_words(data))


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


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

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
lda_model1 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# Print the Keyword in the 20 topics
pprint(lda_model.print_topics())
pprint(lda_model1.print_topics())

# lda = LdaModel(common_corpus, num_topics=20)


doc_lda = lda_model[corpus]

# Save model
topic_model = datapath("model")
lda_model.save(topic_model)