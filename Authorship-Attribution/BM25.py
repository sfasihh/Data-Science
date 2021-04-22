#BM25
import os
import numpy as np
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

path = 'C:\\Users\\samiy\\Desktop\\sam\\collection\\Frankenstein'

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


def PreProcess(final):
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
    
    return final    

def TF_IDF(final):
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
    tf_idf = {}
    TermF = {}
    for i in range(len(final)):
        tokens = final[i]
        for token in np.unique(tokens):
            counter = 0
            for x in tokens:
                if token == x:
                    counter = counter +1
            tf = counter/len(tokens)
            TermF[i, token] = tf
            df = DF[token]
            idf = np.log(len(final)/(df+1))
            tf_idf[i, token] = tf*idf 
    return TermF, tf_idf

def IDF(final, q):
    n=0
    for i in final:
        if q in i:
            n=n+1
    idf = np.log((len(final)-n+0.5)/(n+0.5))
    return idf

def TF(final, D, q):
    TFcounter = 0
    for i in final[D]:
        if q == i:
            TFcounter = TFcounter + 1
    tf = TFcounter/len(final[D])
    return tf

# Q is query
# D is specific document
def BM25(D, Q, final, tf):
    data = []
    data.append(Q)
    tokens = PreProcess(data)
    unique_tokens_query = np.unique(tokens)
    avgdl = 0
    for i in final:
        avgdl = avgdl + len(i)
    avgdl = avgdl/len(final)
    k1 = 1.2
    b=0.75
    
    result = []
    for i in unique_tokens_query:
        result.append(IDF(final, i)*((TF(final, D, i) * (k1 + 1))/(TF(final, D, i) + k1*(1-b+b*(len(final[D])/avgdl)))))
    return sum(result)
                      
    
def Nq(x, w):
    count = 0
    for i in x:
        if w in i:
            count = count + 1
    return count

x = PreProcess(final)
y, z = TF_IDF(x)

title_choice = input("Enter the research paper title: ")
scores = []
for i in range(len(final)):

    scores.append(BM25(i, title_choice, x, y))

print("The most relevant document is " + author_name[scores.index(max(scores))] + " with a score of " + str(max(scores)))
