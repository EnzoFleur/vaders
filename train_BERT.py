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

############# Data ################
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
out_bert = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, truncation=True, padding = 'longest',return_tensors = 'np',return_token_type_ids = False)
D = out_bert['input_ids']
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
    doc_emb,_ = encoder(Xt)
    doc_emb = tf.reshape(doc_emb,[Xt.shape[0],nnn])
    out.append(doc_emb)
   
Xt = D[((i+1)*split)::,:]
doc_emb,_ = encoder(Xt)
doc_emb = tf.reshape(doc_emb,[Xt.shape[0],nnn])
out.append(doc_emb)
doc_emb = np.vstack(out)
print(doc_emb.shape)
         
corpus.add_embedding(dict(zip(id_doc,doc_emb)))
corpus.add_mask(dict(zip(id_doc,out_bert['attention_mask'] )))

import pickle
with open('data/nyt.corpus', 'wb') as f1:
    pickle.dump(corpus, f1)
