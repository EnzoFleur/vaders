#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import sys
import math

import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import tensorflow as tf

import matplotlib.patches as mpatches

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import os
import re
import json
from tqdm import tqdm
from src.corpus import *

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)


############# Data ################
print('Read Corpus')

data_dir = "C:\\Users\\EnzoT\\Documents\\datasets"
dataset = "BlogAuthorshipCorpus"
authors = sorted([a for a in os.listdir(os.path.join(data_dir, dataset)) if os.path.isdir(os.path.join(data_dir, dataset, a))])
documents = []
doc2aut = {}
id_docs = []

for author in tqdm(authors):
    docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, dataset, author))])
    id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

    for doc in docs:
        documents.append(read(os.path.join(data_dir, dataset, author, doc)))
        doc2aut[doc.replace(".txt", "")] = author

aut2id = dict(zip(authors, list(range(len(authors)))))
doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

nd = len(doc2id)
na = len(aut2id)

print("%d documents and %d authors" % (nd,na))

print('Tokenize')
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.tokenizer
tokens = []
m_l = 0
for doc in tqdm(documents):
    ll = [token.text for token in tokenizer(doc.lower()) if len(token.text.strip())>0]
    if m_l < len(ll):
        m_l = len(ll)
    tokens.append(ll)

if m_l > 512:
    m_l=512
    for i, tok in enumerate(tokens):
        tokens[i] = tok[:m_l]

print("Max number of tokens is :", m_l)

print("Building doc embedding")
embeddings_dict = {}
EMBEDDING_SIZE = 300
with open("data/crawl-300d-2M.vec", 'r',encoding="utf-8") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        try:
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
        except:
            print(values)

import pickle

with open(os.path.join('data', dataset, "%s_newvoc.pickle" % dataset), "rb") as ff:
    new_voc=pickle.load(ff)

for w in tqdm(new_voc):
    embeddings_dict[w] = np.zeros((EMBEDDING_SIZE,))

print("Building masks and embeddings")

doc_emb = np.zeros((nd,m_l, EMBEDDING_SIZE))
doc_mask = np.zeros((nd,m_l))
for i,doc in tqdm(enumerate(tokens)):
    tt = []
    ze = 0
    for j,w in enumerate(doc):
        try:
            tt.append(embeddings_dict[w])
            doc_mask[i,j] = 1
            ze += 1
        except:
            tt.append(np.zeros((EMBEDDING_SIZE,)))
    if ze == 0:
        print(doc)
                  
    tt = np.array(tt)
    doc_emb[i,:tt.shape[0],:] = tt

np.save(os.path.join('data', dataset, dataset+'_embds.glove'), doc_emb)
np.save(os.path.join('data', dataset, dataset+'_masks.glove'), doc_mask)

# with open(os.path.join('data', dataset, dataset+'_embds.glove'), 'wb') as f1:
#     pickle.dump(dict(zip(id_docs,doc_emb)), f1)

# with open(os.path.join('data',dataset, dataset+'_masks.glove'), 'wb') as f1:
#     pickle.dump(dict(zip(id_docs,doc_mask)), f1)