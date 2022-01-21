import tensorflow as tf

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

import tensorflow_hub as hub


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
BERT_preprocess_path = "C:\\Users\\EnzoT\\Documents\\code\\bert_preprocess\\"
BERT_preprocessor = hub.load(BERT_preprocess_path)

tokenize = hub.KerasLayer(BERT_preprocessor.tokenize)
tokenized_inputs = [tokenize(doc) for doc in documents[:3]]

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

# tt_DAN = DAN(300, 300)
# tt_D = tf.convert_to_tensor(D[:5,:3,:4], dtype=np.float32)
# tt_Dmask = tf.convert_to_tensor(D_mask[:5,:3], dtype=np.float32)

# tf.random.set_seed(12)
# tt_train=tt_DAN(tt_D, tt_Dmask, training=True)
# tt_eval = tt_DAN(tt_D, tt_Dmask, training=True)
