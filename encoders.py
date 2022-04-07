import numpy as np
import os
import tensorflow as tf
# import tensorflow_text as text

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

import tensorflow_hub as hub

def make_bert_preprocess_model(sentence_features, tfhub_handle_preprocess, seq_length=512):
  """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

  input_segments = [
      tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
      for ft in sentence_features]

  # Tokenize the text to word pieces.
  bert_preprocess = hub.load(tfhub_handle_preprocess)
  tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
  segments = [tokenizer(s) for s in input_segments]

  # Optional: Trim segments in a smart way to fit seq_length.
  # Simple cases (like this example) can skip this step and let
  # the next step apply a default truncation to approximately equal lengths.
  truncated_segments = segments

  # Pack inputs. The details (start/end token ids, dict of output tensors)
  # are model-dependent, so this gets loaded from the SavedModel.
  packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
  model_inputs = packer(truncated_segments)
  return tf.keras.Model(input_segments, model_inputs)

# BERT_preprocess_path = "C:\\Users\\EnzoT\\Documents\\code\\bert_preprocess\\"
# BERT_preprocess = make_bert_preprocess_model(['input1, input2'], BERT_preprocess_path)
BERT_preprocess = None
# BERT_path = "C:\\Users\\EnzoT\\Documents\\code\\bert\\"
# BERT_layer = hub.KerasLayer(BERT_path, trainable = True)
BERT_layer = None

USE_path = os.path.join("..", 'universal-sentence-encoder')
USE_layer = hub.KerasLayer(USE_path, trainable=True)

class MLP(layers.Layer):
    def __init__(self, input_size, output_size, L2=1e-5):
        super(MLP, self).__init__()

        # Network Parameters
        self.input_size = input_size # MNIST data input (img shape: 28*28)
        self.output_size = output_size # MNIST total classes (0-9 digits)

        mlp_layers = [
                layers.Dropout(0.2),
                layers.Dense(self.input_size, kernel_regularizer=tf.keras.regularizers.l2(L2)),
                layers.BatchNormalization(),
                layers.Activation('tanh'),
                layers.Dropout(0.2),
                layers.Dense(self.output_size, kernel_regularizer=tf.keras.regularizers.l2(L2)),
                layers.BatchNormalization(),   
            ]

        self.encoder = tf.keras.Sequential(mlp_layers)

    def call(self, x, training=None):
        
        out_layer = self.encoder(x)

        return out_layer

class DAN(layers.Layer):
    def __init__(self, embedding_dim, n_hidden_units, n_hidden_layers=3, pooling="avg", dropout_prob=0.3, L2=1e-5):
        super(DAN, self).__init__()

        self.pooling = pooling

        self.dropout = layers.Dropout(dropout_prob)
        encoder_layers = []
        for i in range(n_hidden_layers):
            if i==0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            encoder_layers.extend(
                [
                    layers.Dense(input_dim, kernel_regularizer=tf.keras.regularizers.l2(L2)),
                    layers.BatchNormalization(),
                    layers.LeakyReLU(alpha=0.3),
                    layers.Dropout(dropout_prob)
                ]
            )

        self.encoder = tf.keras.Sequential(encoder_layers)

    def _pool(self, embds, masks, batch_size):
        if self.pooling == "avg":
            return tf.math.reduce_sum(embds, axis=1) / tf.reshape(tf.math.reduce_sum(masks, axis=1), (batch_size, 1))
        elif self.pooling == "max":
            emb_max, _ = tf.max(embds, 1)
            return emb_max

    def call(self, inputs, masks, training=None):

        batch_size = inputs.shape[0]

        if training:
            inputs = self.dropout(inputs)
        embds = self._pool(inputs, masks, batch_size)
        outputs = self.encoder(embds)

        return outputs

class TypeGraphConvolution(layers.Layer):
    def __init__(self,in_features, out_features, bias=True):

        super(TypeGraphConvolution, self).__init__()

        self.in_features = in_features 
        self.out_features = out_features

        self.W = tf.Variable(tf.random.normal([self.in_features, self.out_features]), trainable=True)
        if bias:
            self.bias = tf.Variable(tf.random.normal([self.out_features]), trainable=True)
        else:
            self.bias = None

    def call(self, text, adj, dep_embed):

        batch_size, max_len, feat_dim = text.shape
        val_us = tf.expand_dims(text, axis=2)
        val_us = tf.repeat(val_us, max_len, axis=2)

        val_sum = val_us + dep_embed

        adj_us = tf.expand_dims(adj, axis=-1)
        adj_us = tf.repeat(adj_us, feat_dim, axis=-1)
        hidden = tf.linalg.matmul(val_sum, self.W)
        output = tf.transpose(hidden, perm=[0,2,1,3]) * adj_us

        output = tf.math.reduce_sum(output, axis=2)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

def switch_layer(inputs):
    
    inp, emb = inputs
    zeros = tf.zeros_like(inp)
    ones = tf.ones_like(inp)
    
    inp = tf.keras.backend.switch(inp > 0, ones, zeros)
    inp = tf.expand_dims(inp, -1)
    
    return inp * emb

class AsaTgcn(layers.Layer):
    def __init__(self,r,hidden_size, layer_number=3, num_types=47):

        super(AsaTgcn, self).__init__()
    
        self.r = r
        self.hidden_size = hidden_size
        self.layer_number = layer_number
        self.num_types = num_types

        self.TGCNLayers = [TypeGraphConvolution(hidden_size, hidden_size)
                                                for _ in range(self.layer_number)]

        self.batchnorm = layers.BatchNormalization()

        self.fc_single = layers.Dense(self.r, activation=None)
        self.dropout = layers.Dropout(rate=0.1)
        self.ensemble_linear = tf.Variable(tf.ones([3]), trainable=True)

        self.ensemble = tf.Variable(tf.random.normal((self.r, self.r)), trainable=True)
        self.dep_embedding = layers.Embedding(self.num_types, hidden_size, name="dep_embedding", mask_zero=True)
        self.switch = layers.Lambda(switch_layer)

    def get_attention(self, val_out, dep_embed, adj):

        batch_size, max_len, feat_dim = val_out.shape
        val_us = tf.expand_dims(val_out, axis=2)
        val_us = tf.repeat(val_us, [max_len], axis=2)
        val_cat = tf.concat([val_us, dep_embed], axis=-1)
        atten_expand = val_cat * tf.transpose(val_cat, perm=[0, 2, 1, 3])

        attention_score = tf.math.reduce_sum(atten_expand, axis=-1)
        attention_score = attention_score / np.power(feat_dim, 0.5)
        attention_score = tf.math.softmax(attention_score)

        attention_score = tf.math.multiply(attention_score, adj)
        
        return attention_score

    def get_average(self, aspect_indices, x):
        aspect_indices_us = tf.expand_dims(aspect_indices, 2)
        x_mask = x * aspect_indices_us
        aspect_len = tf.math.reduce_sum(tf.cast((aspect_indices_us != 0), tf.float32), axis=1)
        x_sum = tf.math.reduce_sum(x_mask, axis=1)
        x_av = tf.math.divide(x_sum, aspect_len + 1e-10)

        return x_av

    def call(self, text, input_mask, dep_adj_matrix, dep_value_matrix):

        dep_embed = self.dep_embedding(dep_value_matrix)
        dep_embed = self.switch([dep_value_matrix, dep_embed])

        batch_size, max_len, feat_dim = text.shape

        seq_out = self.batchnorm(text)
        seq_out = self.dropout(seq_out)
        attention_score_for_output = []
        tgcn_layers_output = []
        
        for tgcn in self.TGCNLayers:
            attention_score = self.get_attention(seq_out, dep_embed, dep_adj_matrix)
            attention_score_for_output.append(attention_score)
            seq_out = tf.nn.relu(tgcn(seq_out, attention_score, dep_embed))
            tgcn_layers_output.append(seq_out)

        tgcn_layers_output_pool = [self.get_average(input_mask, x_out) for x_out in tgcn_layers_output]

        x_pool = tf.stack(tgcn_layers_output_pool, axis=-1)
        ensemble_out = tf.math.multiply(x_pool, tf.nn.softmax(self.ensemble_linear, axis=0))
        ensemble_out = tf.math.reduce_sum(ensemble_out, axis=-1)
        ensemble_out = self.dropout(ensemble_out)
        output = self.fc_single(ensemble_out)   

        return output

class VADER(tf.keras.Model):
    def __init__(self,nba,r,doc_r,pl, encoder, beta = 1e-12,L=1, loss="CE", alpha=1/2):
    
        super(VADER, self).__init__()
    
        self.loss = loss
        self.alpha = alpha
        self.nba = nba
        self.r = r
        self.doc_r = doc_r
        self.pl = pl
        self.beta = beta
        self.L = L
        self.encoder = encoder

        self.a_authors = tf.Variable(tf.ones([1]),name = 'a_author',trainable = True)
        self.b_authors = tf.Variable(tf.ones([1]),name = 'b_author', trainable = True)       
        
        if self.loss == "CE":
            self.a_features = tf.Variable(tf.ones([1]),name = 'a_features',trainable = True)
            self.b_features = tf.Variable(tf.ones([1]),name = 'b_features', trainable = True)
        # elif self.loss == "L2":
        #     self.l2mlp = MLP(300,300)
  
        if encoder == "DAN":
            self.doc_mean = DAN(self.r, self.r)
            self.doc_var = DAN(self.r, self.r)
        elif encoder == "USE":
            self.doc_encoder = USE_layer
            self.doc_mean = MLP(512, r) 
            self.doc_var =  MLP(512, r)
        elif encoder == "GNN":
            self.doc_encoder = AsaTgcn(self.r, hidden_size=512)
            self.doc_mean = MLP(self.r, self.r)
            self.doc_var = MLP(self.r, self.r)
        elif encoder == "BERT":
            self.doc_encoder = BERT_layer
            self.doc_mean = DAN(768, self.r)
            self.doc_var =  DAN(768, self.r)
        
        self.mean_author = layers.Embedding(self.nba,self.r,tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
                                                name = 'aut_mean')
        
        self.logvar_author = layers.Embedding(self.nba,self.r,tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
                                                name = 'aut_var')
    
    def reparameterize(self, mean, logvar):
        
        eps = tf.random.normal(shape=(mean.shape))
            
        return eps * tf.math.sqrt(tf.math.exp(logvar)) + mean

    def logistic_classifier_features(self, features, doc_emb, apply_sigmoid=True):

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

        if self.encoder == "DAN":
            dmean = self.doc_mean(doc_tok, doc_mask, training=training)
            dvar = self.doc_var(doc_tok, doc_mask, training=training)
        elif self.encoder == "USE":
            doc_emb = self.doc_encoder(doc_tok, training=training)
            dmean = self.doc_mean(doc_emb, training=training)
            dvar = self.doc_var(doc_emb, training=training)
        elif self.encoder == "BERT":
            doc_tok = BERT_preprocess(doc_tok)
            doc_mask = tf.cast(doc_tok['input_mask'], dtype=tf.float32)
            doc_emb = self.doc_encoder(doc_tok, training=training)["sequence_output"]
            dmean = self.doc_mean(doc_emb, doc_mask, training=training)
            dvar = self.doc_var(doc_emb, doc_mask, training=training)
        elif self.encoder == "GNN":
            batch_size, _, seq_len, encode_size = doc_tok.shape

            doc_mask = tf.ones((batch_size, seq_len))
            text = doc_tok[:,0,:,:]
            dep_value_matrix = doc_tok[:,1,:,256:512]
            dep_adj_matrix = doc_tok[:,1,:,:256]

            doc_emb = self.doc_encoder(text, doc_mask, dep_adj_matrix, dep_value_matrix)
            dmean = self.doc_mean(doc_emb, training=training)
            dvar = self.doc_var(doc_emb, training=training)

        return dmean,dvar

def compute_loss(model, documents, pairs, y, yf, training=True):
    
    y_authors, y_features = tf.split(tf.cast(y, dtype=np.float32), [1, y.shape[1] - 1], 1)

    y_authors = tf.squeeze(y_authors)
    y_features = tf.squeeze(y_features)

    i,j = tf.split(pairs, 2, 1)

    if model.encoder == "GNN":
        doc_emb = documents[tf.squeeze(i, axis=1),:,:,:]
    else:
        doc_emb = documents[i][:,0]
    doc_mask = None

    mean_aut = tf.squeeze(model.mean_author(j))
    logvar_aut = tf.squeeze(model.logvar_author(j))

    doc_mean,doc_var = model.encode_doc(doc_emb,doc_mask, training=training)

    ## Soft contrastive #####

    feature_loss = 0
    author_loss = 0

    if model.loss == "L2":
        # Classic and basic bread and butter L2 loss
        feature_loss += model.L * tf.reduce_sum((tf.nn.l2_loss(doc_mean-yf)))

    for draw in range(model.L):

        doc_emb = model.reparameterize(doc_mean, doc_var)
        aut_emb = model.reparameterize(mean_aut, logvar_aut)

        if model.loss == "CE":
            ## Bring closer document embedding and stylistic features
            probs = model.logistic_classifier_features(yf, doc_emb, apply_sigmoid=False)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=y_features)

            feature_loss += tf.reduce_sum(cross_ent)

        ## Bring closer document and author embeddings
        distance = tf.sqrt(tf.reduce_sum(tf.pow(aut_emb - doc_emb, 2), 1))

        probs = model.logistic_classifier(distance, apply_sigmoid=False)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=y_authors)

        author_loss += tf.reduce_sum(cross_ent)
        
    feature_loss *= model.alpha/model.L
    author_loss *= (1-model.alpha)/model.L

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
    
def compute_apply_gradients(model, documents, pairs, y, yf, optimizer):
    
    with tf.GradientTape() as tape:

        loss = compute_loss(model, documents, pairs, y, yf)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))