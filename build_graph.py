import os
import random
import numpy as np
import pickle
from pydantic import EnumError
import scipy.sparse as sp
import sys
from tqdm import tqdm
import re

def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

dataset = "gutenberg"

encoder="GNN"
data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
res_dir = "C:\\Users\\EnzoT\\Documents\\results"

authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
documents = []
doc2aut = {}
id_docs = []

for author in tqdm(authors):
    docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
    id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

    for doc in docs:
        documents.append(read(os.path.join(data_dir, author, doc)))
        doc2aut[doc.replace(".txt", "")] = author

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", model_max_length=512)
model = TFAutoModel.from_pretrained("bert-base-cased", output_hidden_states=False)

import spacy
dep_tags = ['self', 'ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']
dep_to_id = {k: i+1 for i, k in enumerate(dep_tags)}

nlp = spacy.load("en_core_web_sm")

max_len=512
full_x_emb = []
full_adj_mat = []
full_dep_adj_mat = []
for book in tqdm(documents):
    doc = nlp(book)
    encoded = tokenizer.encode_plus(book, return_tensors="tf", truncation=True, max_length=max_len, padding=True)
    output = tf.squeeze(model(**encoded).last_hidden_state)


    tok_to_id = {tok:i for i, tok in enumerate(doc)}

    adj_matrix = np.zeros((len(doc), len(doc)))
    dep_adj_matrix = np.zeros((len(doc), len(doc)))
    x_emb = []
    roots=[]
    for i, token in enumerate(doc):
        if not i in encoded.word_ids():
            break

        token_ids_word = np.where(np.array(encoded.word_ids()) == i)
        x_emb.append(tf.math.reduce_mean(tf.gather(output, np.where(np.array(encoded.word_ids()) == i)[0]),0))
        
        if token.dep_ == "ROOT":
            for root in roots:
                adj_matrix[tok_to_id[token], tok_to_id[root]]=1
                dep_adj_matrix[tok_to_id[token], tok_to_id[root]]=dep_to_id['ROOT']
            roots.append(token)

        for child in token.children:
            adj_matrix[tok_to_id[token], tok_to_id[child]]=1
            dep_adj_matrix[tok_to_id[token], tok_to_id[child]]=dep_to_id[child.dep_]

    adj_matrix=adj_matrix[:i,:i]
    dep_adj_matrix = dep_adj_matrix[:i,:i]

    adj_matrix = adj_matrix + adj_matrix.T + np.diag(np.ones(i))
    dep_adj_matrix = dep_adj_matrix + dep_adj_matrix.T + np.diag(np.ones(i))

    x_emb = tf.stack(x_emb, axis=0)

    full_x_emb.append(x_emb)
    full_adj_mat.append(adj_matrix)
    full_dep_adj_mat.append(dep_adj_matrix)

full_x_emb = tf.stack(x_emb).to_numpy()
full_adj_mat = tf.stack(adj_matrix).to_numpy()
full_dep_adj_mat = tf.stack(dep_adj_matrix).numpy()

np.save("%s\\text.npy" % dataset, x_emb)
np.save("%s\\dep_adj_matrix.npy" % dataset, full_adj_mat)
np.save("%s\\dep_value_matrix.npy" % dataset, full_dep_adj_mat)