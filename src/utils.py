#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.sparse as sp
import json
import time

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize

def tf2idf(tf):
    
    idf = tf.copy()
    idf[idf.nonzero()] = 1
    
    num = np.zeros((1,tf.shape[1])) + tf.shape[0]
    denom =  np.sum(idf,axis = 0)
    
    m_idf = np.log(num/denom)
    
    idf = idf.multiply(m_idf)
    
    idf.eliminate_zeros()
    
    idf = tf.multiply(idf)
    
    return idf

#aut2tf, tf,tf_idf,voc,aut_doc,aut2map= aut2tf(Corpus_name) 
#tf_idf = tf2idf(aut2tf)

def sparse2bow(tf):
    bo = [[] for i in range(tf.shape[0])]
    tf = tf.tocoo()
    for i,j,v in zip(tf.row, tf.col, tf.data):
        bo[i].append((int(j),int(v)))
    return bo

def bow2numpy(tf,shape0,shape1):
    out = np.zeros((shape0,shape1))
    for i in range(shape0):
        row = tf[i]
        for tup in row:
            out[i,tup[0]] = tup[1]
    return out

def sparse2dict(tf):
    bo = {}
    for i in range(tf.shape[0]):
        bo[i] = []
    tf = tf.tocoo()
    for i,j,v in zip(tf.row, tf.col, tf.data):
        bo[i].append(int(j))  
    return bo

def author_raw(aut_doc,id2doc,raw_docs):
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    raw = [tokenizer.tokenize(i.lower()) for i in raw_docs]
        
    naut = aut_doc.shape[0]
    
    rows,cols = aut_doc.nonzero()
    
    aut_raw = [[] for i in range(naut)]
                
    for i in range(naut):
        doceu = aut_doc[i,:].nonzero()[1].flatten()
        raw_aut = [raw[j] for j in doceu]
        aut_raw[i] = [item for sublist in raw_aut for item in sublist]

    return aut_raw

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

def eval_svm(embeddings, labels, ratio, C, verbose=True):
    d = embeddings.shape[1]
    scores = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=ratio, random_state=i)
        classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=C, multi_class='ovr', fit_intercept=True, class_weight=None, random_state=i, max_iter=4000)
        classifier.fit(X_train, y_train)
        y_pred = []
        for x in X_test:
            x = x.reshape((1,d))
            y_pred.append(classifier.predict(x))
        y_pred = np.asarray(y_pred)
        scores.append(accuracy_score(y_test, y_pred)*100)

    accuracy_mean = np.mean(scores)
    accuracy_std = np.std(scores)
   
    if verbose:
        print('Accuracy (%.1f): %.3f (std: %.3f)' % (ratio, accuracy_mean, accuracy_std))
    return accuracy_mean, accuracy_std

def find_optimal_C(embeddings, labels):
    Cs = [1, 2, 4, 8]
    scores = [eval_svm(embeddings, labels, 0.5, C, verbose=False) for C in Cs]
    accuracy_scores = [accuracy_mean for accuracy_mean, accuracy_std in scores]
    return Cs[accuracy_scores.index(max(accuracy_scores))]

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def doc2v(txt,d,dm = 0,epoch=5):
    
        print("Learning document embeddings from given data in dimension %d " % d)
        
        tokenizer = RegexpTokenizer(r'\w+')
    
        raw = [tokenizer.tokenize(i.lower()) for i in txt]
        
        documents = []
        for j in range(len(raw)):
            documents.append(TaggedDocument(raw[j], [j]))
            
        #print(documents)

        model = Doc2Vec(vector_size=d,dm = dm)
            
        model.build_vocab(documents)
        
        model.train(documents, total_examples=model.corpus_count, epochs=epoch)
        
        ndoc = len(txt)
        D = np.zeros((ndoc, d))
        
        for i in range(ndoc):
            D[i,:] = model.docvecs[i]
            
        return D

