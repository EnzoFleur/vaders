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
    
    doc = tokenizer.encode(doc, add_special_tokens=False)

    parts = chunks(doc, (maxLength - 2))
    parts = parts[:4]
    parts = [[bertStartIndex] + part + [bertEndIndex] for part in parts]

    if len(parts) > 1 and len(parts[-1]) < int(maxLength * multiSamplageMinMaxLengthRatio):
        parts = parts[:-1]

    # We pad the last part:
    parts[-1] = parts[-1] + [0] * (maxLength - len(parts[-1]))
    # We check the length of each part:
    for part in parts:
        assert len(part) == maxLength

    return parts

class DeepStyle(tf.keras.Model):
    def __init__(self,nba):
    
        super(DeepStyle, self).__init__()
    
        self.nba = nba

        self.encoder = TFDistilBertModel.from_pretrained(os.path.join("..","distilBERT", "distilbert-base-uncased"), local_files_only=True)
        self.dropout = layers.Dropout(0.1)
        self.classifier = layers.Dense(nba)
        
    def encode_doc(self, text, part_mask):
        batch_size, n_parts, r = text.shape
        hidden_full = []

        for b in range(batch_size):
            mask = tf.reduce_sum(part_mask[b])
            hidden_state = self.encoder(text[b,:tf.cast(mask, np.int32),:])[0][:,0]
            hidden_full.append(tf.math.reduce_sum(hidden_state, axis=0)/tf.cast(mask, np.float32))

        return tf.stack(hidden_full)

    def encode_author(self, text, part_mask):
        batch_size, n_parts, r = text.shape
        hidden_full = []

        for b in range(batch_size):
            mask = tf.reduce_sum(part_mask[b])
            hidden_state = self.encoder(text[b,:tf.cast(mask, np.int32),:])[0][:,0]
            hidden_full.append(tf.math.reduce_sum(hidden_state, axis=0)/tf.cast(mask, np.float32))

        return tf.math.reduce_mean(tf.stack(hidden_full), axis=0)

    def call(self, text, part_mask, training=None):

        batch_size, n_parts, r = text.shape
        hidden_full = []

        for b in range(batch_size):
            mask = tf.reduce_sum(part_mask[b])
            hidden_state = self.encoder(text[b,:tf.cast(mask, np.int32),:])[0][:,0]
            hidden_full.append(tf.math.reduce_sum(hidden_state, axis=0)/tf.cast(mask, np.float32))

        output = self.dropout(tf.stack(hidden_full))

        output = self.classifier(output)

        return output

def compute_loss(model, documents, part_mask, pairs):
    
    i,j = tf.split(pairs, 2, 1)
    i = tf.squeeze(i, axis=1)
    y_true = tf.one_hot(tf.squeeze(j, axis=1), depth = model.nba)

    doc_tok = tf.convert_to_tensor(documents[i.numpy()], dtype=np.int32)
    mask_tok = tf.convert_to_tensor(part_mask[i.numpy()], dtype=np.int32)

    y_pred = model(doc_tok, mask_tok, training=True)
    loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

    return loss
    
def compute_apply_gradients(model, documents, part_mask, pairs, optimizer):
    
    with tf.GradientTape() as tape:

        loss = compute_loss(model, documents, part_mask, pairs)
        
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
    args = parser.parse_args()

    data_dir = args.dataset
    epochs = args.epochs
    batch_size = args.batchsize
    negpairs = args.negpairs

    method = "deep_style_%d" % epochs

    ############# Data ################
    # data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"
    # batch_size = 128
    # epochs=100

    authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
    encoded_doc = []
    doc2aut = {}
    id_docs = []
    part_mask = []

    for author in tqdm(authors):
        docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])
        id_docs = [*id_docs, *[doc.replace(".txt", "") for doc in docs]]

        for doc in docs:
            doc2aut[doc.replace(".txt", "")] = author
            content = read(os.path.join(data_dir, author, doc))
            tf_content = tf.convert_to_tensor(distilBertEncode(content))
            part_mask.append(len(tf_content))
            encoded_doc.append(tf_content)
            
    aut2id = dict(zip(authors, list(range(len(authors)))))
    doc2id = dict(zip(id_docs, list(range(len(id_docs)))))

    nd = len(doc2id)
    na = len(aut2id)
    
    maxparts = max([len(i) for i in encoded_doc])

    tf_documents = np.zeros((nd, maxparts, 512), dtype=np.int32)

    for i, (tens, part) in enumerate(zip(encoded_doc, part_mask)):
        tf_documents[i,:part,:] = tens.numpy()

    part_mask = [[*[1]*i,*[0]*(maxparts-i)] for i in part_mask]
    part_mask = np.array(part_mask, dtype=np.int32)

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

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model = DeepStyle(na) 

    result = []
    pairs = next(iter(train_data))

    print("Training the model")
    for epoch in range(1, epochs + 1):

        f_loss = compute_loss(model, tf_documents, part_mask, pairs)
        print("[%d/%d]  F-loss : %.3f" % (epoch, epochs, f_loss), flush=True)
        
        for pairs in tqdm(train_data):
            compute_apply_gradients(model, tf_documents, part_mask, pairs, optimizer)

        if epoch % 5 == 0:
            aut_emb = []
            for i in range(model.nba):
                books = np.array(di2ai_df_train[di2ai_df_train.authors==i].documents)
                doc_tok = tf.convert_to_tensor(tf_documents[books], dtype=np.int32)
                mask_tok = tf.convert_to_tensor(part_mask[books], dtype=np.int32)
                aut_emb.append(model.encode_author(doc_tok, mask_tok).numpy()) 

            aut_emb = np.vstack(aut_emb)

            split = 256
            nb = int(len(doc_tp) / split )
            out= []
            for i in tqdm(range(nb)): 
                start = (i*split ) 
                stop = start + split
                doc_tok = tf.convert_to_tensor(tf_documents[doc_tp[start:stop]], dtype=np.int32)
                mask_tok = tf.convert_to_tensor(part_mask[doc_tp[start:stop]], dtype=np.int32)
                doc_emb = model.encode_doc(doc_tok, mask_tok) 
                out.append(doc_emb)

            doc_tok = tf.convert_to_tensor(tf_documents[doc_tp[((i+1)*split)::]], dtype=np.int32)
            mask_tok = tf.convert_to_tensor(part_mask[doc_tp[((i+1)*split)::]], dtype=np.int32)
            doc_emb,_ = model.encode_doc(doc_tok, mask_tok)                                 
            out.append(doc_emb)
            doc_emb = np.vstack(out)

            print("Evaluation Aut id")

            aa = normalize(aut_emb, axis=1)
            dd = normalize(doc_emb[np.sort(doc_tp)], axis=1)
            y_score = normalize( dd @ aa.transpose(),norm="l1")
            ce = coverage_error(aut_doc_test[doc_tp,:], y_score)
            lr = label_ranking_average_precision_score(aut_doc_test[doc_tp,:], y_score)*100
            print("coverage, precision")
            print(str(round(ce,2)) + ", "+ str(round(lr,2)))

    if not os.path.isdir(os.path.join("results",method)):
        os.mkdir(os.path.join("results",method))
 
    features = pd.read_csv(os.path.join("data", "gutenberg", "features", "features.csv"), sep=";")
    res_df = style_embedding_evaluation(aut_emb, features.groupby("author").mean().reset_index(), n_fold=10)
    res_df.to_csv(os.path.join("results", method, "style_%s.csv" % method), sep=";")
