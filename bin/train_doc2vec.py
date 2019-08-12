import numpy as np
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.downloader as api
from pattern.en import lemma
from scipy.spatial.distance import cosine


vocab = set(words.words())

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
for df in pd.read_csv("/opt/data/train.csv", usecols=["question1","question2","is_duplicate"], chunksize=1000):
    df.fillna(value="", inplace=True)
    docs = []
    for enum, (v1, v2, target) in enumerate(zip(df.question1, df.question2, df.is_duplicate)):
        v1 = clean_and_lemmatize_doc(v1, True)
        v1 = " ".join(v1)
        v2 = clean_and_lemmatize_doc(v2, True)
        v2 = " ".join(v2)
        if target == "":
            target = 0
        docs.append(v1 + "," + v2 + "," + str(target))
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


## train on wiki
doc_num = 0
wiki = api.load('wiki-english-20171001')
docs = []
for enum,doc in enumerate(wiki):
    doc = " ".join(doc['section_texts'])
    doc = clean_and_lemmatize_doc(doc, False)
    docs.append(TaggedDocument(doc, [doc_num]))
    doc_num += 1
    if (enum % 100):
        print("doc num: " + str(enum*100))
        model.train(docs, total_examples=len(docs), epochs=1)
        model.save('/opt/model.h5')
        docs = []

        

## train on local data
epochs = range(1)
for epoch in epochs:
    for chunk_num, df in enumerate(pd.read_csv("/opt/data/cleaned_docs.csv", names=["question1","question2","is_duplicate"], chunksize=1000)):
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






## create final dataframe of doc2vec value and total cosine similarity
for df in pd.read_csv("/opt/data/train.csv", usecols=['question1', 'question2', 'is_duplicate'], chunk_size=1000)
    new_df = []
    for enum, (col1, col2, target) in enumerate(zip(df.question1, df.question2, df.is_duplicate)):
        v1 = list(model.infer_vector(clean_and_lemmatize_doc(v1, False)))
        v2 = list(model.infer_vector(clean_and_lemmatize_doc(v2, False)))
        v = list(np.subtract(v1, v2))
        v = v + [cosine(v1,v2), target]
        new_df.append(v)
        if ((enum % 100 == 0) & (enum!=0)) | (enum == len(df)-1):
            print("row number: " + str(enum))
            with open('/opt/final_vectors.csv', 'a') as f:
                f.write("\n".join(new_df) + "\n")
                new_df = []


