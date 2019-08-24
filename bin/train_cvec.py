import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from pattern.en import lemma
from scipy.spatial.distance import cosine, euclidean
from keras.activations import softmax
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import callbacks


vocab = {}
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
                    if cleaned_word in vocab.keys():
                        vocab[cleaned_word] += 1
                    else:
                        vocab[cleaned_word] = 0
        except:
            continue
    return stripped_doc


stop_words = ['so', 'now', 'just', 'that','then','to','and','or','i','he','she','his','her','my','for','by','at','of','while','as','beacuse','if','but','the','a','an','do','does','doing','did','having','has','had','am','is','are','be','who','whom','this','their','it','itself','is','herself','hers','hiself','himself','his','him','you','your','yours','yourself','youre','our','ourself','ourselves','i','me','my','myself','we']
stop_words = ["".join(clean_and_lemmatize_doc(word, False)) for word in stop_words]
stop_words = set(stop_words)


## clean local data and write to disk as a string
counter = 0
for filename in ['train.csv']:
    for df in pd.read_csv("/opt/data/" + filename, usecols=["question1","question2","is_duplicate"], chunksize=1000):
        df.dropna(inplace=True)
        docs = []
        for enum, (v1, v2, target) in enumerate(zip(df.question1, df.question2, df.is_duplicate)):
            v1 = clean_and_lemmatize_doc(v1, True)
            v1 = " ".join(v1)
            v2 = clean_and_lemmatize_doc(v2, True)
            v2 = " ".join(v2)
            docs.append(v1 + "," + v2 + "," + str(target))
            counter += 1
            if ((counter % 100 == 0) & (counter!=0)) | (enum == len(df)-1):
                print("row number: " + str(counter))
                with open('/opt/data/cleaned_docs.csv', 'a') as f:
                    f.write("\n".join(docs) + "\n")
                    docs = []

vocab_list = list(vocab.keys())
for key in vocab_list:
    if vocab[key] < 2:
        del vocab[key]

vocab = sorted(list(vocab.keys()))
vocab_list = []

## save vocab
with open('/opt/data/vocab.txt', 'w') as f:
    f.write(" ".join(vocab))

def word_share(x):
    x = x.split("--*delimiter*--")
    x1 = x[0].split()
    x2 = x[1].split()
    return len(list(set(x1) & set(x2))) / float(len(set(x1 + x2)))

def one_hot(x):
    if x>0:
        return 1
    else:
        return 0

def get_doc_len_diff(x):
    x = x.split("--*delimiter*--")
    x1 = len(x[0].split())
    x2 = len(x[1].split())
    return abs(x1-x2)

def get_doc_len_diff_pct(x):
    x = x.split("--*delimiter*--")
    x1 = len(x[0].split())
    x2 = len(x[1].split())
    longest_doc_len = sorted([x1, x2])[-1]
    return abs(x1-x2) / float(longest_doc_len)


## initiate model
input_dim=5
model = Sequential()
model.add(Dense(input_dim*2, input_dim=input_dim, activation='relu'))
model.add(Dense(input_dim, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



## initialize cvec model
counter = 0
cvec = CountVectorizer(stop_words=stop_words, ngram_range=(0,1), analyzer='word', vocabulary=vocab)
for df in pd.read_csv('/opt/data/cleaned_docs.csv', names=["question1","question2", "is_duplicate"], chunksize=1000):
    df.dropna(inplace=True)
    print("row num: " + str(counter * 1000))
    counter += 1
    if counter < 10:
        epochs = 15
    else:
        epochs = 1
    df['concatenated'] = df.question1 + "--*delimiter*--" + df.question2    
    df['word_share_pct'] = df.concatenated.apply(lambda x: word_share(x))
    df['doc_len_diff'] = df.concatenated.apply(lambda x: get_doc_len_diff(x))
    df['doc_len_diff_pct'] = df.concatenated.apply(lambda x: get_doc_len_diff_pct(x))
    q1 = pd.DataFrame(cvec.fit_transform(df.question1).todense(), columns=cvec.get_feature_names())
    q2 = pd.DataFrame(cvec.fit_transform(df.question2).todense(), columns=cvec.get_feature_names())
    final = []
    for idx in q1.index:
        v1 = q1.loc[idx] / q1.loc[idx].sum()
        v2 = q2.loc[idx] / q2.loc[idx].sum()
        cos = cosine(v1, v2)
        v1 = q1.loc[idx]
        v1 = [one_hot(x) for x in v1]
        v2 = q2.loc[idx] 
        v2 = [one_hot(x) for x in v2]
        cos2 = cosine(v1, v2)
        if (cos == float('nan')) | (cos2 == float('nan')):
            continue
        final.append([cos, cos2])
    final = pd.DataFrame(final, columns=['cosine1', 'cosine2'])
    final['word_share_pct'] = list(df.word_share_pct)
    final['doc_len_diff'] = list(df.doc_len_diff)
    final['doc_len_diff_pct'] = list(df.doc_len_diff_pct)
    final['target'] = list(df.is_duplicate)
    final.dropna(inplace=True)
    x_train = final.drop(['target'],axis=1).values
    y_train = np_utils.to_categorical(final.target.values,2)
    df = []
    model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=1000, verbose=2)
    model.save("/opt/cvec_model.h5")
