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
dataset = "Lyrics"
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


print('Read Corpus')   
corpus = Corpus.read_data("data/nyt")
print("%d documents and %d authors" % (corpus.nd,corpus.na))

aut2id = dict(zip([*corpus.id2aut.values()],[*corpus.id2aut]))
id_doc = [*corpus.docs]
doc2id = dict(zip(id_doc,list(range(len(id_doc)))))


print('get content')
sentences = corpus.get_text()


from transformers import BertTokenizer,TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
out_bert = tokenizer.batch_encode_plus(documents, add_special_tokens=True, truncation=True, padding = 'longest',return_tensors = 'np',return_token_type_ids = False)
D = out_bert['input_ids']
D_mask = out_bert['attention_mask']
print(D.shape)
max_l = D.shape[1]

encoder = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)

print("building doc embedding")
n = D.shape[0]
split = 100
nb = int(n / split )
print(nb)
out= []
nnn = 768 * max_l
iout = 0
for i in tqdm(range(nb)): 
    start = (i*split ) 
    stop = start + split 
    Xt = D[start:stop,:]
    doc_emb = encoder(Xt).last_hidden_state
    # doc_emb = tf.reshape(doc_emb,[Xt.shape[0],nnn])
    out.append(doc_emb)
   
Xt = D[((i+1)*split)::,:]
doc_emb = encoder(Xt).last_hidden_state
# doc_emb = tf.reshape(doc_emb,[Xt.shape[0],nnn])
out.append(doc_emb)
doc_emb = np.vstack(out)
print(doc_emb.shape)

np.save(os.path.join('data', dataset, dataset+'_embds.bert'), doc_emb)
np.save(os.path.join('data', dataset, dataset+'_masks.bert'), D_mask)