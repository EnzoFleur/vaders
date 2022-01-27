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

# tt_DAN = DAN(300, 300)
# tt_D = tf.convert_to_tensor(D[:5,:3,:4], dtype=np.float32)
# tt_Dmask = tf.convert_to_tensor(D_mask[:5,:3], dtype=np.float32)

# tf.random.set_seed(12)
# tt_train=tt_DAN(tt_D, tt_Dmask, training=True)
# tt_eval = tt_DAN(tt_D, tt_Dmask, training=True)
