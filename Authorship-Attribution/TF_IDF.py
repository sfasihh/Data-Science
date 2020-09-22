#TF/ID
import annoy
from annoy import AnnoyIndex
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
import pickle
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

path = 'C:\\Users\\samiy\\Desktop\\collection'

files = []
final = []
# r=root, d=directories, f = files
#Loading all of the Files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
for f in files:
    temp=open(f, 'r')
    try:
        final.append(temp.read().strip())
        temp.close()
    except:
        temp.close()
        continue
print(final)
# Pre-processing
# Finding Unique Words
# Getting rid of Punctuation
# Lemmetiziation
stop_words = stopwords.words('english')
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for x in range(len(final)):
    final[x] = final[x].lower()
    final[x] = final[x].split(' ')
    new_text = []
    for y in range(len(final[x])):
        if final[x][y] not in stop_words:
            temp = nlp(final[x][y])
            for i in temp:
                new_text.append(i.lemma_)
    final[x] = new_text
    for i in symbols:
        if len(final[x]) != 0:
            final[x] = np.char.replace(final[x], i, '')
            final[x] = list(final[x])
            


for j in range(len(final)):
    for i in final[j]:
        if i == '':
            final[j].remove('')
#Finding DF, TF, and tf_idf
# DF = Document frequency. # of times word appears in all document
# TF = Term Frequency. # of times a word appears in single document divided by number of words in single document
# idf = Inverse document frequency. # of words in all documents divided by Document frequency of a specific word. Then logged. 
# tf_idf = Tf multiply idf. Larger value means word is more specific to a single document. Lower value means word exists throughout all documents. 
DF = {}
for i in range(len(final)):
    tokens = final[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

for i in DF:
    DF[i] = len(DF[i])

with open('DF.pickle', 'wb') as f:
    pickle.dump(DF, f)

tf_idf = {}
for i in range(len(final)):
    tokens = final[i]
    for token in np.unique(tokens):
        counter = 0
        for x in tokens:
            if token == x:
                counter = counter +1
        tf = counter/len(tokens)
        df = DF[token]
        idf = np.log(len(final)/(df+1))
        tf_idf[i, token] = tf*idf


# Building annoy index. Need to create a vectors of x length for each author in directory. X is the number of unique words found in all documents. 
index = AnnoyIndex(len(DF))
counter=-1
for i in final:
    v = []
    counter=counter+1
    for j in DF:
        if j in i:
            v.append(tf_idf[counter, j])
        else:
            v.append(-100)

    index.add_item(counter, v)
        
index.build(1000)
index.save('TF_IDF.ann')
print("Saved")
