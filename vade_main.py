#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from datetime import datetime

import os
import re
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
 
from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error,label_ranking_average_precision_score, accuracy_score

from encoders import VADER, compute_apply_gradients, compute_loss
from regressor import style_embedding_evaluation

# import random

# os.environ['TF_CUDNN_DETERMINISTIC']='1'
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-o','--outdir', type=str,
                        help='Path to output directory')
    parser.add_argument('-e','--encoder', default="USE", type=str,
                        help='Document encoder to use')
    parser.add_argument('-b','--beta', default=1e-12, type=float,
                        help='Beta parameter value')
    parser.add_argument('-bs','--batchsize', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('-a','--alpha', default=1/2, type=float,
                        help='Alpha parameter value')
    parser.add_argument('-l','--loss', default="CE", type=str,
                        help='Type of feature loss (L2 or CE)')
    parser.add_argument('-n','--negpairs', default=1, type=int,
                        help='Number of negative pairs to sample')
    parser.add_argument('-lr','--learningrate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('-s','--surname', default='', type=str,
                        help='name')
    args = parser.parse_args()

    data_dir = args.dataset
    res_dir=args.outdir
    dataset = data_dir.split(os.sep)[-1]
    beta = args.beta
    epochs = args.epochs
    batch_size = args.batchsize
    name=args.surname

    encoder = args.encoder
    alpha = args.alpha
    negpairs = args.negpairs
    loss = args.loss
    lrate = args.learningrate

    # ############ Data ################
    # dataset = "gutenberg"

    # encoder="USE"
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # res_dir = "C:\\Users\\EnzoT\\Documents\\results"
    # beta=1e-12
    # alpha=1/2
    # loss="CE"
    # negpairs = 10
    # batch_size = 128
    # epochs=100
    # lrate=1e-3
    # name="features"

    if lrate == 1e-3:
        method = "%s_%s_%s_%6f_%3f_%d_%s" % (loss,encoder, dataset, beta, alpha, negpairs, name)
    else:
        method = "LR01_%s_%s_%s_%6f_%3f_%d_%s" % (loss,encoder, dataset, beta, alpha, negpairs, name)
        
    if not os.path.isdir(os.path.join("results", method)):
        os.mkdir(os.path.join("results", method))

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
    documents = []
    doc2aut = {}
    id_docs = []
    # s_authors =[28,32,33,35,64,92,99,100,101,112,
    #     114,116,125,127,144,153,154,162,164,168,
    #     169,170,174,179,183,196,197,202,220,225,
    #     235,236,241,242,243,246,251,255,259,271,
    #     278,290,293,299,300,302,310,329,330,338,
    #     343,344,348,356,359,367,368,371,378,379,
    #     381,399,402,404,405,410,416,422,425,426,
    #     432,448,456,481,482,489,498,509,516,517,
    #     525,538,539,549,562,584,590,597,601,602,
    #     605,606,613,614,624,631,640,643,655,662]
    # authors= [authors[i] for i in s_authors]

    for author in tqdm(authors):
        docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
        id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

        for doc in docs:
            doc2aut[doc.replace(".txt", "")] = author
            if not (encoder=="GNN"):
                documents.append(read(os.path.join(data_dir, author, doc)))

    aut2id = dict(zip(authors, list(range(len(authors)))))
    doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

    nd = len(doc2id)
    na = len(aut2id)

    if encoder == "GNN":
        documents = np.zeros((nd, 2, 256, 512)).astype(np.float32)
        documents[:,0,:,:] = np.load(os.path.join("data", dataset, "text.npy")).astype(np.float32)
        documents[:,1,:,:256] = np.load(os.path.join("data", dataset, "dep_adj_matrix.npy")).astype(np.float32)
        documents[:,1,:,256:512] = np.load(os.path.join("data", dataset, "dep_value_matrix.npy")).astype(np.float32)
    else:
        documents = np.array(documents)

    di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

    print('Get features')
    features = pd.read_csv(os.path.join("data", dataset, "features", "features.csv"), sep=";")
    features = features.drop(["author", 'needn\'t', 'couldn\'t', 'hasn\'t', 'mightn\'t', 'you\'ve', 'shan\'t', 'aren',
        'weren\'t', 'mustn', 'shan', 'should\'ve', 'mightn', 'needn', 'hadn\'t',
        'aren\'t', 'hadn', 'that\'ll', '£', '€', '<', '\'', '^', '~'], axis=1).replace({"id":{k+".txt":v for k,v in doc2id.items()}}).sort_values("id", ascending=True)
    features = np.array(features.sort_values("id",ascending=True).drop("id", axis=1))
    stdScale = StandardScaler()
    features = stdScale.fit_transform(features)

    # If features encoding is needed
    # documents = features

    print("Build pairs")
    di2ai_df = pd.DataFrame([di2ai.keys(), di2ai.values()], index=['documents','authors']).T
    di2ai_df_train, di2ai_test = train_test_split(di2ai_df, test_size = 0.2, stratify = di2ai_df['authors'])

    # For testing purpose
    doc_tp = np.sort(list(di2ai_test.documents))
    aut_doc_test = np.array(pd.crosstab(di2ai_df.documents, di2ai_df.authors).sort_values(by='documents', ascending=True))

    features_train = []
    data_pairs = []
    labels = []

    for d, a in di2ai_df_train.itertuples(index=False, name=None):
        # True author, true features
        data_pairs.append((d,a))
        features_train.append(features[d])
        labels.append([1,1])


        # # True author, wrong features
        data_pairs.extend([(d,a) for _ in range(negpairs)])
        features_train.extend([features[di2ai_df_train[di2ai_df_train.documents != d].documents.sample(1).values[0]] for _ in range(negpairs)])
        labels.extend([[1,0] for _ in range(negpairs)])

        # Wrong author, true features
        data_pairs.extend(zip([d]*negpairs, di2ai_df_train[di2ai_df_train.authors!=a].authors.sample(negpairs)))
        features_train.extend([features[d] for _ in range(negpairs)])
        labels.extend([[0,1] for _ in range(negpairs)])

        # # Wrong author, wrong features
        data_pairs.extend(zip([d]*negpairs, di2ai_df_train[di2ai_df_train.authors!=a].authors.sample(negpairs)))
        features_train.extend([features[di2ai_df_train[di2ai_df_train.documents != d].documents.sample(1).values[0]] for _ in range(negpairs)])
        labels.extend([[0,0] for _ in range(negpairs)])

    features_train = np.float32(np.array(features_train))

    print("%d documents and %d authors\n%d data pairs for training" % (nd,na, len(data_pairs)))

    r = 300
    doc_r = r
    max_l = 512

    print("Embedding in dimension %d, padding in %d" % (r,max_l))

    ############ Splitting Data #########

    train_data = tf.data.Dataset.from_tensor_slices((data_pairs,features_train,labels)).shuffle(len(labels)).batch(batch_size)

    ############# Training ################

    print("Building the model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)

    model = VADER(na,r,doc_r,max_l, encoder=encoder, beta=beta, L=10, alpha=alpha, loss=loss) 

    def optimizer_custom_decay(optimizer, step):
        schedule = [1e-3, 5e-4, 2e-4, 5e-5, 3e-5, 1e-5]
        #get the optimizer configuration dictionary
        opt_cfg = optimizer.get_config() 

        #change the value of learning rate by multiplying decay rate with learning rate to get new learning rate
        opt_cfg['learning_rate'] = opt_cfg['learning_rate']/opt_cfg['learning_rate'] * schedule[min(step, len(schedule))]
        print("Reducing learning rate to %f" % opt_cfg['learning_rate'], flush=True)

        optimizer = optimizer.from_config(opt_cfg)
        return optimizer
            
    result = []
    pairs, yf, y = next(iter(train_data))

    val_loss = 0.00
    step=0
    memory = []

    print("Training the model")
    for epoch in range(1, epochs + 1):
        
        start_time = datetime.now()
        f_losses, a_losses, i_losses = 0,0,0
        for pairs, yf, y in tqdm(train_data):
            f_loss, a_loss, i_loss = compute_apply_gradients(model, documents, pairs, y, yf, optimizer)
            f_losses += f_loss
            a_losses += a_loss
            i_losses += i_loss

        print("[%d/%d] in %s F-loss : %.3f | A-loss : %.3f | I-loss : %.3f" % (epoch, epochs, str(datetime.now()-start_time), f_losses/len(pairs), a_loss/len(pairs), i_loss/len(pairs)), flush=True)

        if epoch % 1 == 0:
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
                Xt = documents[start:stop]
                doc_emb,_ = model.encode_doc(Xt,None, training=False) 
                out.append(doc_emb)    
            Xt = documents[((i+1)*split)::]
            doc_emb,_ = model.encode_doc(Xt,None, training=False)                                 
            out.append(doc_emb)
            doc_emb = np.vstack(out)

            print("Evaluation Aut id")

            aa = normalize(aut_emb, axis=1)
            dd = normalize(doc_emb[np.sort(doc_tp)], axis=1)
            y_score = normalize( dd @ aa.transpose(),norm="l1")
            ce = coverage_error(aut_doc_test[doc_tp,:], y_score)/na*100
            lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
            ac = accuracy_score(np.argmax(aut_doc_test[doc_tp], axis=1), np.argmax(y_score, axis=1)) *100
            print("coverage, precision, accuracy", flush=True)
            print(str(round(ce,2)) + ", "+ str(round(lr,2)) + ", "+ str(round(ac,2)))
            result.append(ac)

            memory.append(ac)
            val_loss = max(memory)
            if val_loss not in memory[-2:]:
                step+=1
                optimizer = optimizer_custom_decay(optimizer, step)

            # model.save_weights(os.path.join("results", method, "%s.ckpt" % method))

            np.save(os.path.join("results", method, "aut_%s_%d.npy" % (method, epoch)), aut_emb)
            np.save(os.path.join("results", method, "aut_var_%s_%d.npy" % (method, epoch)), aut_var)
            np.save(os.path.join("results", method, "doc_%s_%d.npy" % (method, epoch)), doc_emb)

    with open(os.path.join("results", method, 'res_%s.txt' % method), 'w') as f:
        for item in result:
            f.write("%s\n" % item)

    print("Building author and doc embedding")    
    aut_emb = []
    aut_var = []
    for i in range(model.nba):
        aut_emb.append(np.asarray(model.mean_author(i)))  
        aut_var.append(np.asarray(tf.math.exp(model.logvar_author(i))))
    aut_emb = np.vstack(aut_emb)
    aut_var = np.vstack(aut_var)

    split = 256
    nb = int(nd / split )
    out= []
    for i in tqdm(range(nb)): 
        start = (i*split ) 
        stop = start + split 
        Xt = documents[start:stop]
        doc_emb,_ = model.encode_doc(Xt,None, training=False) 
        out.append(doc_emb)    
    Xt = documents[((i+1)*split)::]
    doc_emb,_ = model.encode_doc(Xt,None, training=False)                                 
    out.append(doc_emb)
    doc_emb = np.vstack(out)

    #################################################### Eval ##################################################
    print("Evaluation Aut id")
    y_score = normalize(normalize(doc_emb[doc_tp], axis=1) @ normalize(aut_emb, axis=1).transpose(),norm="l1")
    ce = (coverage_error(aut_doc_test[doc_tp,:], y_score))/na*100
    lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
    ac = accuracy_score(np.argmax(aut_doc_test[doc_tp], axis=1), np.argmax(y_score, axis=1)) *100

    print("coverage, precision, accuracy")
    print(str(round(ce,2)) + ", "+ str(round(lr,2)) + ", "+ str(round(ac,2)))
    with open(os.path.join("results", method, "coverage_%s.txt" % method), "a+") as f:
        f.write(method+" & "+str(round(ce,2)) + " & "+ str(round(lr,2)) + " & "+ str(round(ac,2)) + "\\\ \n")

    np.save(os.path.join("results", method, "aut_%s.npy" % method), aut_emb)
    np.save(os.path.join("results", method, "aut_var_%s.npy" % method), aut_var)
    np.save(os.path.join("results", method, "doc_%s.npy" % method), doc_emb)

    ################################################### Style Eval ##############################################

    features = pd.read_csv(os.path.join("data", dataset, "features", "features.csv"), sep=";")
    res_df = style_embedding_evaluation(aut_emb, features.groupby("author").mean().reset_index(), n_fold=10)
    res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")
    print(res_df)
    # res_df = style_embedding_evaluation(doc_embd, features.drop(['author', 'id'], axis=1), n_fold=2)
