import numpy as np
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.downloader as api
from pattern.en import lemma
from scipy.spatial.distance import cosine, euclidean


vocab = set(words.words())
vocab = ["".join(clean_and_lemmatize_doc(word, False)) for word in vocab]
vocab = set(vocab)

def clean_and_lemmatize_doc(doc, build_vocab):
    stripped_doc = []
    for char in doc:
        if (ord(char) >= 48) & (ord(char) <= 57):
            stripped_doc.append(char)
        elif (ord(char) >= 65) & (ord(char) <= 89):
            stripped_doc.append(char)
        elif (ord(char) >= 97) & (ord(char) <= 122):
            stripped_doc.append(char)
        elif ord(char) == 39:
            continue
        else:
            stripped_doc.append(" ")
    doc = "".join(stripped_doc).lower()
    doc = doc.split(" ")
    stripped_doc = []
    for word in doc:
        try:
            cleaned_word = lemma(word)
            if len(cleaned_word) > 0:
                stripped_doc.append(cleaned_word)
                if build_vocab == True:
                    vocab.add(cleaned_word)
        except:
            continue
    return stripped_doc


## clean local data and write to disk as a string
counter = 0
for filename in ['train.csv', 'test.csv']:
    for df in pd.read_csv("/opt/data/" + filename, usecols=["question1","question2"], chunksize=1000):
        df.dropna(inplace=True)
        docs = []
        for enum, (v1, v2) in enumerate(zip(df.question1, df.question2)):
            v1 = clean_and_lemmatize_doc(v1, True)
            v1 = " ".join(v1)
            v2 = clean_and_lemmatize_doc(v2, True)
            v2 = " ".join(v2)
            docs.append(v1 + "," + v2)
            counter += 1
            if ((counter % 100 == 0) & (counter!=0)) | (enum == len(df)-1):
                print("row number: " + str(counter))
                with open('/opt/data/cleaned_docs.csv', 'a') as f:
                    f.write("\n".join(docs))
                    docs = []

## save vocab
with open('/opt/data/vocab.txt', 'w') as f:
    f.write(" ".join(vocab))

## vocab len for 2vec model
vocab_len = len(vocab)


## initialize Doc2Vec model
model = Doc2Vec(vector_size=300, min_count=1, workers=4, dbow_words=1, epochs=1)
model.build_vocab(corpus_file='/opt/data/vocab.txt')

doc_num = 0

"""
## train on corpus
corpus = api.load('wiki-english-20171001')
docs = []
for enum,doc in enumerate(corpus):
    doc = " ".join(doc['section_texts'])
    doc = clean_and_lemmatize_doc(doc, False)
    docs.append(TaggedDocument(doc, [doc_num]))
    doc_num += 1
    if (enum % 100):
        print("doc num: " + str(enum*100))
        model.train(docs, total_examples=len(docs), epochs=1)
        model.save('/opt/model.h5')
        docs = []
        

## train on news
corpus = api.load('20-newsgroups')
docs = []
for enum,doc in enumerate(corpus):
    doc = " ".join(doc['data'])
    doc = clean_and_lemmatize_doc(doc, False)
    docs.append(TaggedDocument(doc, [doc_num]))
    doc_num += 1
    if (enum % 100):
        print("doc num: " + str(enum*100))
        model.train(docs, total_examples=len(docs), epochs=1)
        model.save('/opt/model.h5')
        docs = []
"""


## train on local data
epochs = range(1)
for epoch in epochs:
    for chunk_num, df in enumerate(pd.read_csv("/opt/data/cleaned_docs.csv", names=["question1","question2"], chunksize=1000)):
        print("chunk: " + str(chunk_num))
        docs = []
        for col in ['question1', 'question2']:
            for doc in df[col]:
                doc_num += 1
                doc = doc.split(" ")
                docs.append(TaggedDocument(doc, [doc_num]))
        ## train batch
        model.train(docs, total_examples=len(docs), epochs=1)
        model.save('/opt/model.h5')




stop_words = ['so', 'now', 'just', 'that','then','to','and','or','i','he','she','his','her','my','for','by','at','of','while','as','beacuse','if','but','the','a','an','do','does','doing','did','having','has','had','am','is','are','be','who','whom','this','their','it','itself','is','herself','hers','hiself','himself','his','him','you','your','yours','yourself','youre','our','ourself','ourselves','i','me','my','myself','we']
stop_words = ["".join(clean_and_lemmatize_doc(word, False)) for word in stop_words]
stop_words = set(stop_words)


## create final dataframe of doc2vec value and total cosine similarity
for df in pd.read_csv("/opt/data/train.csv", usecols=['question1', 'question2', 'is_duplicate'], chunksize=1000):
    new_df = []
    for enum, (v1, v2, target) in enumerate(zip(df.question1, df.question2, df.is_duplicate)):
        try:
            v1 = clean_and_lemmatize_doc(v1, False)
            v2 = clean_and_lemmatize_doc(v2, False)
            doc_len_diff = abs(len(v1) - len(v2))
            longest_doc_len = sorted([len(v1), len(v2)])[-1]
            doc_len_diff_pct = doc_len_diff / float(longest_doc_len)
            v1_stop = [x for x in v1 if x not in stop_words]
            v2_stop = [x for x in v2 if x not in stop_words]
            share_pct = len(set([x for x in v1_stop if x in v2_stop])) / float(len(set(v1_stop + v2_stop)))
            cos = cosine(list(model.infer_vector(v1)), list(model.infer_vector(v2)))
            euclid = euclidean(list(model.infer_vector(v1)), list(model.infer_vector(v2)))
            v =  [str(doc_len_diff), str(doc_len_diff_pct), str(share_pct), str(euclid), str(target)]
            new_df.append(",".join(v))
            if ((enum % 100 == 0) & (enum!=0)) | (enum == len(df)-1):
                print("row number: " + str(enum))
                with open('/opt/data/final_vectors.csv', 'a') as f:
                    f.write("\n".join(new_df) + "\n")
                    new_df = []
        except:
            print("ERROR")
            print(v1)
            print(v2)
            print("")


