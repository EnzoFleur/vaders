import numpy as np
import os
import tensorflow as tf
import re
import argparse

from tensorflow.keras import layers,Model, activations
from tensorflow.keras.initializers import Constant

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error,label_ranking_average_precision_score

import pandas as pd
from tqdm import tqdm

from regressor import style_embedding_evaluation
from transformers import DistilBertTokenizer, TFDistilBertModel

def chunks(*args, **kwargs):
	return list(chunksYielder(*args, **kwargs))
def chunksYielder(l, n):
	"""Yield successive n-sized chunks from l."""
	if l is None:
		return []
	for i in range(0, len(l), n):
		yield l[i:i + n]

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

def distilBertEncode(doc, maxLength=512, multiSamplageMinMaxLengthRatio=0.3, bertStartIndex=101, bertEndIndex=102):
    tokenizer = DistilBertTokenizer.from_pretrained(os.path.join("..","distilBERT", "distilbert-base-uncased"), local_files_only=True)
    
    doc = tokenizer.encode(doc, add_special_tokens=True, truncation=True, max_length=512, padding=True)

    return doc

class DeepStyle(tf.keras.Model):
    def __init__(self,nba):
    
        super(DeepStyle, self).__init__()
    
        self.nba = nba

        self.encoder = TFDistilBertModel.from_pretrained(os.path.join("..","distilBERT", "distilbert-base-uncased"), local_files_only=True)
        self.dropout = layers.Dropout(0.1)
        self.classifier = layers.Dense(nba)
        
    def encode_doc(self, text):
        hidden_state = self.encoder(text)[0][:,0]

        return hidden_state

    def encode_author(self, text):
        hidden_state = self.encoder(text)[0][:,0]

        return tf.math.reduce_mean((hidden_state), axis=0)

    def call(self, text, training=None):

        hidden_state = self.encoder(text)[0][:,0]

        output = self.dropout(hidden_state)

        output = self.classifier(output)

        return output

def compute_loss(model, documents, mask, pairs):
    
    i,j = tf.split(pairs, 2, 1)
    i = tf.squeeze(i, axis=1)
    y_true = tf.one_hot(tf.squeeze(j, axis=1), depth = model.nba)

    doc_tok = {'input_ids':documents[i.numpy()], 'attention_mask': mask[i.numpy()]}

    y_pred = model(doc_tok, training=True)
    loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

    return loss
    
def compute_apply_gradients(model, documents, mask, pairs, optimizer):
    
    with tf.GradientTape() as tape:

        loss = compute_loss(model, documents, mask, pairs)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-bs','--batchsize', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('-n','--negpairs', default=1, type=int,
                        help='Number of negative pairs to sample')
    parser.add_argument('-s','--surname', default='', type=str,
                        help='name')
    args = parser.parse_args()

    data_dir = args.dataset
    epochs = args.epochs
    batch_size = args.batchsize
    negpairs = args.negpairs
    name=args.surname

    method = "deep_style_%s" % name

    # ############# Data ################
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # batch_size = 5
    # epochs=5
    # negpairs=2
    # name='slip'

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
    documents = []
    doc2aut = {}
    id_docs = []
    part_mask = []

    for author in tqdm(authors[:10]):
        docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
        id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

        for doc in docs:
            doc2aut[doc.replace(".txt", "")] = author
            content = read(os.path.join(data_dir, author, doc))
            # tf_content = tf.convert_to_tensor(distilBertEncode(content))
            documents.append(content)

    tokenizer = DistilBertTokenizer.from_pretrained(os.path.join("..","distilBERT", "distilbert-base-uncased"), local_files_only=True)
    
    documents = tokenizer(documents, add_special_tokens=True, truncation=True, max_length=512, padding='longest')

    mask = np.vstack(documents.attention_mask)
    documents = np.vstack(documents.input_ids)

    aut2id = dict(zip(authors, list(range(len(authors)))))
    doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

    nd = len(doc2id)
    na = len(aut2id)
    
    documents = np.vstack(documents)

    di2ai = {doc2id[d]: aut2id[a] for d,a in doc2aut.items()}

    print("Build pairs")
    di2ai_df = pd.DataFrame([di2ai.keys(), di2ai.values()], index=['documents','authors']).T
    di2ai_df_train, di2ai_test = train_test_split(di2ai_df, test_size = 0.2, stratify = di2ai_df['authors'])

    # For testing purpose
    doc_tp = np.sort(list(di2ai_test.documents))
    aut_doc_test = np.array(pd.crosstab(di2ai_df.documents, di2ai_df.authors).sort_values(by='documents', ascending=True))

    data_pairs = []

    for d, a in di2ai_df_train.itertuples(index=False, name=None):
        # True author, true features
        data_pairs.append((d,a))

        # Wrong author, true features
        data_pairs.extend(zip([d]*negpairs, di2ai_df_train[di2ai_df_train.authors!=a].authors.sample(negpairs)))

    train_data = tf.data.Dataset.from_tensor_slices(data_pairs).shuffle(len(data_pairs)).batch(batch_size)

    ############# Training ################

    print("Building the model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model = DeepStyle(na) 

    result = []
    pairs = next(iter(train_data))
    features = pd.read_csv(os.path.join("data", "gutenberg", "features", "features.csv"), sep=";")

    print("Training the model")
    for epoch in range(1, epochs + 1):

        if epoch > 5:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        if epoch > 15:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

        f_loss = compute_loss(model, documents, mask, pairs)
        print("[%d/%d]  F-loss : %.3f" % (epoch, epochs, f_loss), flush=True)
        
        for pairs in tqdm(train_data):
            compute_apply_gradients(model, documents, mask, pairs, optimizer)

        if epoch % 5 == 0:
            aut_emb = []
            for i in range(model.nba):
                books = np.array(di2ai_df_train[di2ai_df_train.authors==i].documents)
                doc_tok = {'input_ids':documents[books], 'attention_mask': mask[books]}
                aut_emb.append(model.encode_author(doc_tok).numpy()) 

            aut_emb = np.vstack(aut_emb)

            split = 256
            nb = int(len(doc_tp) / split )
            out= []
            for i in tqdm(range(nb)): 
                start = (i*split ) 
                stop = start + split
                doc_tok = {'input_ids':documents[doc_tp[start:stop]], 'attention_mask': mask[doc_tp[start:stop]]}
                doc_emb = model.encode_doc(doc_tok) 
                out.append(doc_emb)

            doc_tok = {'input_ids':documents[doc_tp[((i+1)*split)::]], 'attention_mask': mask[doc_tp[((i+1)*split)::]]}
            doc_emb = model.encode_doc(doc_tok)                                
            out.append(doc_emb)
            doc_emb = np.vstack(out)

            print("Evaluation Aut id")

            aa = normalize(aut_emb, axis=1)
            dd = normalize(doc_emb, axis=1)
            y_score = normalize( dd @ aa.transpose(),norm="l1")
            ce = coverage_error(aut_doc_test[doc_tp,:], y_score)
            lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
            print("coverage, precision")
            print(str(round(ce,2)) + ", "+ str(round(lr,2)))

        if epoch % 10 == 0:
            res_df = style_embedding_evaluation(aut_emb, features.groupby("author").mean().reset_index(), n_fold=10)
            print(res_df)
    if not os.path.isdir(os.path.join("results",method)):
        os.mkdir(os.path.join("results",method))
 
    res_df = style_embedding_evaluation(aut_emb, features.groupby("author").mean().reset_index(), n_fold=10)
    res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")
