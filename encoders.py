import tensorflow as tf

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant

import tensorflow_hub as hub

USE_layer = hub.KerasLayer("C:\\Users\\EnzoT\\Documents\\code\\universal-sentence-encoder\\", trainable=True)

class MLP(layers.Layer):
    def __init__(self, input_size, output_size, n_hidden_1=256, n_hidden_2=256):
        # Network Parameters
        self.n_hidden_1 = n_hidden_1 # 1st layer number of neurons
        self.n_hidden_2 = n_hidden_2 # 2nd layer number of neurons
        self.input_size = input_size # MNIST data input (img shape: 28*28)
        self.output_size = output_size # MNIST total classes (0-9 digits)

        # Store layers weight & bias
        self.h1 = tf.Variable(tf.random_normal([self.input_size, self.n_hidden_1]), trainable=True)
        self.h2 =  tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), trainable=True)
        self.out = tf.Variable(tf.random_normal([self.n_hidden_2, self.output_size]), trainable=True)

        self.b1 = tf.Variable(tf.random_normal([n_hidden_1]), trainable=True)
        self.b2 = tf.Variable(tf.random_normal([n_hidden_2]), trainable=True)
        self.bout = tf.Variable(tf.random_normal([self.output_size]), trainable=True)


    def call(self, x):
        
        layer_1 = tf.math.tanh(tf.add(tf.matmul(x, self.h1), self.b1))
        layer_2 = tf.math.tanh(tf.add(tf.matmul(layer_1, self.h2), self.b2))
        out_layer = tf.matmul(layer_2, self.out) + self.bout

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
                    layers.Dense(n_hidden_units, kernel_regularizer=tf.keras.regularizers.l2(L2)),
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
