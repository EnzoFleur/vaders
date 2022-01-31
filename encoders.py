import numpy as np
import tensorflow as tf
import tensorflow_text as text

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

BERT_preprocess_path = "C:\\Users\\EnzoT\\Documents\\code\\bert_preprocess\\"
BERT_preprocess = make_bert_preprocess_model(['input1, input2'], BERT_preprocess_path)

BERT_path = "C:\\Users\\EnzoT\\Documents\\code\\bert\\"
BERT_layer = hub.KerasLayer(BERT_path, trainable = True)

USE_path = "C:\\Users\\EnzoT\\Documents\\code\\universal-sentence-encoder\\"
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


class VADER(tf.keras.Model):
    def __init__(self,nba,r,doc_r,pl, encoder, beta = 1e-12,L=1):
    
        super(VADER, self).__init__()
    
        self.nba = nba
        self.r = r
        self.doc_r = doc_r
        self.pl = pl
        self.beta = beta
        self.L = L
        self.encoder = encoder

        self.a_authors = tf.Variable(tf.ones([1]),name = 'a_author',trainable = True)
        self.b_authors = tf.Variable(tf.ones([1]),name = 'b_author', trainable = True)       
        
        self.a_features = tf.Variable(tf.ones([1]),name = 'a_features',trainable = True)
        self.b_features = tf.Variable(tf.ones([1]),name = 'b_features', trainable = True)   

        if encoder == "DAN":
            self.doc_mean = DAN(self.r, self.r)
            self.doc_var = DAN(self.r, self.r)
        elif encoder == "USE":
            self.doc_encoder = USE_layer
            self.doc_mean = MLP(512, r) 
            self.doc_var =  MLP(512, r)
        elif encoder == "GNN":
            pass
        elif encoder == "BERT":
            self.doc_encoder = BERT_layer
            self.doc_mean = DAN(768, self.r)
            self.doc_var =  DAN(768, self.r)
        
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

        if self.encoder == "DAN":
            doc_mean = self.doc_mean(doc_tok, doc_mask, training=training)
            doc_var = self.doc_var(doc_tok, doc_mask, training=training)
        elif self.encoder == "USE":
            doc_emb = self.doc_encoder(doc_tok, training=training)
            doc_mean = self.doc_mean(doc_emb, training=training)
            doc_var = self.doc_var(doc_emb, training=training)
        elif self.encoder == "BERT":
            doc_tok = BERT_preprocess(doc_tok)
            doc_mask = tf.cast(doc_tok['input_mask'], dtype=tf.float32)
            doc_emb = self.doc_encoder(doc_tok, training=training)["sequence_output"]
            doc_mean = self.doc_mean(doc_emb, doc_mask, training=training)
            doc_var = self.doc_var(doc_emb, doc_mask, training=training)

        return doc_mean,doc_var

def compute_loss(model, documents, pairs, y, yf, training=True):
    
    y_authors, y_features = tf.split(tf.cast(y, dtype=np.float32), [1, y.shape[1] - 1], 1)

    y_authors = tf.squeeze(y_authors)
    y_features = tf.squeeze(y_features)

    i,j = tf.split(pairs, 2, 1)

    doc_emb = documents[i][:,0]
    doc_mask = None

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
    
def compute_apply_gradients(model, documents, pairs, y, yf, optimizer):
    
    with tf.GradientTape() as tape:

        loss = compute_loss(model, documents, pairs, y, yf)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# tt_DAN = DAN(300, 300)
# tt_D = tf.convert_to_tensor(D[:5,:3,:4], dtype=np.float32)
# tt_Dmask = tf.convert_to_tensor(D_mask[:5,:3], dtype=np.float32)

# tf.random.set_seed(12)
# tt_train=tt_DAN(tt_D, tt_Dmask, training=True)
# tt_eval = tt_DAN(tt_D, tt_Dmask, training=True)
