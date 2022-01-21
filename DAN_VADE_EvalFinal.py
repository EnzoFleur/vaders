#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import sys
import math
import os
import re
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.patches as mpatches
 
from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error,label_ranking_average_precision_score

from encoders import DAN

############# Data ################
dataset = "gutenberg"
method = "DAN-VADE-GLOVE"

data_dir = "C:\\Users\\EnzoT\\Documents\\datasets"
res_dir = "C:\\Users\\EnzoT\\Documents\\results"

authors = sorted([a for a in os.listdir(os.path.join(data_dir, dataset)) if os.path.isdir(os.path.join(data_dir, dataset, a))])
documents = []
doc2aut = {}
id_docs = []

for author in tqdm(authors):
    docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, dataset, author))])
    id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]
    for doc in docs:
        doc2aut[doc.replace(".txt", "")] = author

aut2id = dict(zip(authors, list(range(len(authors)))))
doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

nd = len(doc2id)
na = len(aut2id)

di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

print('Get features')
features = pd.read_csv(os.path.join(res_dir, dataset, "features", "features.csv"), sep=";").drop(["author", 'needn\'t', 'couldn\'t', 'hasn\'t', 'mightn\'t', 'you\'ve', 'shan\'t', 'aren',
       'weren\'t', 'mustn', 'shan', 'should\'ve', 'mightn', 'needn', 'hadn\'t',
       'aren\'t', 'hadn', 'that\'ll', '£', '€', '<', '\'', '^', '~'], axis=1).replace({"id":{k+".txt":v for k,v in doc2id.items()}}).sort_values("id", ascending=True)
features = np.array(features.sort_values("id",ascending=True).drop("id", axis=1))
stdScale = StandardScaler()
features = stdScale.fit_transform(features)

print("Build pairs")
di2ai_df = pd.DataFrame([di2ai.keys(), di2ai.values()], index=['documents','authors']).T
di2ai_df_train, di2ai_test = train_test_split(di2ai_df, test_size = 0.2, stratify = di2ai_df['authors'])

# For testing purpose
doc_tp = np.sort(list(di2ai_test.documents))
aut_doc_test = np.array(pd.crosstab(di2ai_df.documents, di2ai_df.authors).sort_values(by='documents', ascending=True))

# features_distance = distance_matrix(features, features, p=2)
features_train = []
data_pairs = []
labels = []

for d, a in di2ai_df_train.itertuples(index=False, name=None):
    # ind = np.argpartition(features_distance[d,:], -50)[-50:]

    # True author, true features
    data_pairs.append((d,a))
    features_train.append(features[d])
    labels.append([1,1])

    # # True author, wrong features
    data_pairs.append((d, a))
    features_train.append(features[di2ai_df_train[di2ai_df_train.documents != d].documents.sample().values[0]])
    labels.append([1,0])

    # Wrong author, true features
    data_pairs.append((d, di2ai_df_train[di2ai_df_train.authors!=a].authors.sample(1).values[0]))
    features_train.append(features[d])
    labels.append([0,1])

    # # Wrong author, wrong features
    data_pairs.append((d, di2ai_df_train[di2ai_df_train.authors!=a].authors.sample(1).values[0]))
    features_train.append(features[di2ai_df_train[di2ai_df_train.documents != d].documents.sample().values[0]])
    labels.append([0,0])

features_train = np.float32(np.array(features_train))

print("%d documents and %d authors\n%d data pairs for training" % (nd,na, len(data_pairs)))
########################################################################################################################################

D = np.load(os.path.join("data", dataset, dataset + "_embds.glove.npy")).astype(np.float32)
D_mask = np.load(os.path.join("data", dataset, dataset + "_masks.glove.npy")).astype(np.float32)
# with open(os.path.join("data", dataset, dataset + "_embds.glove"), 'rb') as ff:
#     D = pickle.load(ff)

# D = np.array(list(D.values()), dtype=np.float32)

# with open(os.path.join("data", dataset, dataset + "_masks.glove"), 'rb') as ff:
#     D_mask = pickle.load(ff)

# D_mask = np.array(list(D_mask.values()), dtype=np.float32)

r = 300
doc_r = r
max_l = int(D.shape[1])

print("Embedding in dimension %d, padding in %d" % (r,max_l))

############ Splitting Data #########
batch_size = 64

train_data = tf.data.Dataset.from_tensor_slices((data_pairs,features_train,labels)).shuffle(len(labels)).batch(batch_size)

class VADER(tf.keras.Model):
    def __init__(self,nba,r,doc_r,pl, beta = 1e-12,L=5):
      
        super(VADER, self).__init__()
      
        self.nba = nba
        self.r = r
        self.doc_r = doc_r
        self.pl = pl
        self.beta = beta
        self.L = L

        self.a_authors = tf.Variable(tf.ones([1]),name = 'a_author',trainable = True)
        self.b_authors = tf.Variable(tf.ones([1]),name = 'b_author', trainable = True)       
        
        self.a_features = tf.Variable(tf.ones([1]),name = 'a_features',trainable = True)
        self.b_features = tf.Variable(tf.ones([1]),name = 'b_features', trainable = True)   

        self.doc_mean = DAN(self.r, self.r)
        self.doc_var = DAN(self.r, self.r)
        
        self.mean_author = layers.Embedding(self.nba,self.r,tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
                                                 name = 'aut_mean')
        
        self.logvar_author = layers.Embedding(self.nba,self.r,tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
                                                 name = 'aut_var')
    
    def compute_distance(self,doc_mean,doc_var,mean,var):

        aut_emb = self.reparameterize(mean,var)
        doc_emb = self.reparameterize(doc_mean,doc_var)

        return tf.sqrt(tf.reduce_sum(tf.pow(aut_emb - doc_emb, 2), 1))     

    def reparameterize(self, mean, logvar):
        
        eps = tf.random.normal(shape=(mean.shape))
            
        return eps * tf.math.sqrt(tf.math.exp(logvar)) + mean

    def logistic_classifier_features(self, features, mean, var, apply_sigmoid=True):

        doc_emb = self.reparameterize(mean, var)

        distance = tf.sqrt(tf.reduce_sum(tf.pow(features - doc_emb, 2), 1))

        logits = tf.math.add(tf.multiply(-tf.math.exp(self.a_features),distance),self.b_features)

        if apply_sigmoid:
            logits = tf.sigmoid(logits)
            
        return logits

    def logistic_classifier(self, x, apply_sigmoid=True):

        logits = tf.math.add(tf.multiply(-tf.math.exp(self.a_authors),x),self.b_authors)

        if apply_sigmoid:
            logits = tf.sigmoid(logits)
            
        return logits

    def encode_doc(self,doc_tok,doc_mask, training=True):

        doc_mean = self.doc_mean(doc_tok, doc_mask, training=training)
        doc_var = self.doc_var(doc_tok, doc_mask, training=training)

        return doc_mean,doc_var

def compute_loss(model, D, D_mask, pairs, y, yf, training=True):
    
    y_authors, y_features = tf.split(tf.cast(y, dtype=np.float32), [1, y.shape[1] - 1], 1)

    y_authors = tf.squeeze(y_authors)
    y_features = tf.squeeze(y_features)

    i,j = tf.split(pairs, 2, 1)

    doc_emb = tf.squeeze(D[i], axis=1)
    doc_mask = tf.squeeze(D_mask[i], axis=1)

    mean_aut = tf.squeeze(model.mean_author(j))
    logvar_aut = tf.squeeze(model.logvar_author(j))

    doc_mean,doc_var = model.encode_doc(doc_emb,doc_mask, training=training)

    ## Soft contrastive #####

    feature_loss = 0
    author_loss = 0

    for draw in range(model.L):

        ## Bring closer document embedding and stylistic features
        probs = model.logistic_classifier_features(yf, doc_mean, doc_var, apply_sigmoid=False)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=y_features)

        feature_loss += tf.reduce_sum(cross_ent)

        ## Bring closer document and author embeddings
        distance = model.compute_distance(doc_mean,doc_var,mean_aut,logvar_aut)

        probs = model.logistic_classifier(distance, apply_sigmoid=False)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=y_authors)

        author_loss += tf.reduce_sum(cross_ent)
        
    feature_loss /= model.L
    author_loss /= model.L
    
    ## Info Loss ####

    KL_loss_aut_emb = 0.5 * tf.reduce_sum(tf.square(mean_aut)
                                      + tf.math.exp(logvar_aut) - logvar_aut - 1)
    KL_loss_aut_emb += 0.5 * tf.reduce_sum(tf.square(doc_mean)
                                      + tf.math.exp(doc_var) - doc_var - 1)

    BETA = model.beta

    info_loss = BETA * KL_loss_aut_emb

    if not training:
        return feature_loss, author_loss, info_loss

    return feature_loss + author_loss + info_loss
	
def compute_apply_gradients(model, D, D_mask, pairs, y, yf, optimizer):
    
    with tf.GradientTape() as tape:

        loss = compute_loss(model, D, D_mask, pairs, y, yf)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

############# Training ################

print("Building the model")

r = doc_r
epochs = 500
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = VADER(na,r,doc_r,max_l, beta=1e-12, L=5) 

result = []
pairs, yf, y =next(iter(train_data))

print("Training the model")
for epoch in range(1, epochs + 1):

    f_loss, a_loss, i_loss = compute_loss(model, D, D_mask, pairs, y, yf, training=False)
    print("[%d/%d]  F-loss : %.3f | A-loss : %.3f | I-loss : %.3f" % (epoch, epochs, f_loss, a_loss, i_loss), flush=True)
    
    start_time = time.time()
    for pairs, yf, y in tqdm(train_data):
        compute_apply_gradients(model, D, D_mask, pairs, y, yf, optimizer)
    end_time = time.time()

    if epoch % 5 == 0:
        aut_emb = []
        for i in range(model.nba):
            aut_emb.append(np.asarray(model.mean_author(i)))   
        aut_emb = np.vstack(aut_emb)

        aut_var = []
        for i in range(model.nba):
            aut_var.append(np.asarray(model.logvar_author(i)))   
        aut_var = np.vstack(aut_var)
        
        split = 256
        nb = int(nd / split )
        out= []
        for i in tqdm(range(nb)): 
            start = (i*split ) 
            stop = start + split
            Xt = D[start:stop,:]
            Xt_mask = D_mask[start:stop,:]
            doc_emb,_ = model.encode_doc(Xt,Xt_mask, training=False) 
            out.append(doc_emb)    
        Xt = D[((i+1)*split)::,:]
        Xt_mask = D_mask[((i+1)*split)::,:]
        doc_emb,_ = model.encode_doc(Xt,Xt_mask, training=False)                                 
        out.append(doc_emb)
        doc_emb = np.vstack(out)

        print("Evaluation Aut id")

        aa = normalize(aut_emb, axis=1)
        dd = normalize(doc_emb[np.sort(doc_tp)], axis=1)
        y_score = normalize( dd @ aa.transpose(),norm="l1")
        ce = coverage_error(aut_doc_test[doc_tp,:], y_score)
        print("coverage Cosine",flush=True)
        print(str(round(ce,2)))
        result.append(ce)

with open('res_rep.txt', 'w') as f:
    for item in result:
        f.write("%s\n" % item)

print("Building author and doc embedding")    
aut_emb = []
for i in range(model.nba):
    aut_emb.append(np.asarray(model.mean_author(i)))   
aut_emb = np.vstack(aut_emb)

split = 256
nb = int(nd / split )
out= []
for i in tqdm(range(nb)): 
    start = (i*split ) 
    stop = start + split 
    Xt = D[start:stop,:]
    Xt_mask = D_mask[start:stop,:]
    doc_emb,_ = model.encode_doc(Xt,Xt_mask, training=False) 
    out.append(doc_emb)    
Xt = D[((i+1)*split)::,:]
Xt_mask = D_mask[((i+1)*split)::,:]
doc_emb,_ = model.encode_doc(Xt,Xt_mask)                                 
out.append(doc_emb)
doc_emb = np.vstack(out)

#################################################### Eval ##################################################
print("Evaluation Aut id")
y_score = normalize(normalize(doc_emb[doc_tp], axis=1) @ normalize(aut_emb, axis=1).transpose(),norm="l1")
ce = (coverage_error(aut_doc_test[doc_tp,:], y_score)/na)*100
lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
print("coverage, precision")
print(str(round(ce,2)) + ", "+ str(round(lr,2)))
output = open("coverage_"+dataset+".txt", "a+")
output.write(method+" & "+str(round(ce,2)) + " & "+ str(round(lr,2)) + "\\\ \n")
output.close()

np.save("aut_embds_b0_gutenberg_vanilla.npy", aut_emb)
np.save("doc_embds_b0_gutenberg_vanilla.npy", doc_emb)


################################################### Style Eval ##############################################

features = pd.read_csv(os.path.join(res_dir, dataset, "features", "features.csv"), sep=";")
res_df = style_embedding_evaluation(aut_emb, features.groupby("author").mean().reset_index(), n_fold=10)
res_df = style_embedding_evaluation(doc_embds, features.drop(['author', 'id'], axis=1), n_fold=2)
