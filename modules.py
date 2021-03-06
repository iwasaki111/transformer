import tensorflow as tf
from tensorflow.keras.layers import Lambda
import numpy as np
from keras import backend as K


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        params_shape = input_shape[-1:]
        self.beta = self.add_weight(shape=params_shape, name='beta', initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=params_shape, name='gamma', initializer='ones', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = {"epsilon": float(self.epsilon)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_units, zero_pad=True, scale=True, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zero_pad = zero_pad
        self.scale = scale

    def build(self, input_shape):
        self.lookup_table = self.add_weight(name='lookup', shape=[self.vocab_size, self.num_units],
                                            initializer=tf.keras.initializers.glorot_uniform(), trainable=True,
                                            dtype='float32')
        if self.zero_pad:
            self.lookup_table = tf.concat((tf.zeros(shape=[1, self.num_units]),
                                           self.lookup_table[1:, :]), 0)

    def call(self, inputs, **kwargs):
        outputs = tf.nn.embedding_lookup(self.lookup_table, inputs)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1] + [self.vocab_size])

    def get_config(self):
        config = {'vocab_size': float(self.vocab_size),
                  'num_units': float(self.num_units),
                  'zero_pad': float(self.zero_pad),
                  'scale': float(self.scale),
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def multi_head_attention(queries, keys, prefix, num_units=512, num_heads=4, causality=False, dropout_rate=0.1):
    def split_concat(x):
        return tf.concat(tf.split(x, num_heads, axis=2), axis=0)

    _, T, C = queries.get_shape().as_list()
    T = -1 if T is None else T
    head_unit_dim = num_units // num_heads

    # Linear projections
    Q = tf.keras.layers.Dense(num_units, activation='relu', name=prefix + 'q_fc')(queries)
    K = tf.keras.layers.Dense(num_units, activation='relu', name=prefix + 'k_fc')(keys)
    V = tf.keras.layers.Dense(num_units, activation='relu', name=prefix + 'v_fc')(keys)

    # Split and concat
    Q_ = Lambda(split_concat, output_shape=(-1, T, head_unit_dim), name=prefix + 'split_concat_q')(Q)  # (h*N, T_q, C/h)
    K_ = Lambda(split_concat, output_shape=(-1, T, head_unit_dim), name=prefix + 'split_concat_k')(K)  # (h*N, T_q, C/h)
    V_ = Lambda(split_concat, output_shape=(-1, T, head_unit_dim), name=prefix + 'split_concat_v')(V)  # (h*N, T_q, C/h)

    # Multiplication
    outputs = Lambda(lambda x: tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])),
                     output_shape=(-1, T, T), name=prefix + 'Multiplication')([Q_, K_])

    # Scale
    outputs = Lambda(lambda x: x / (head_unit_dim ** 0.5), name=prefix + 'Scale')(outputs)

    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = Lambda(lambda x: tf.where(tf.equal(key_masks, 0), paddings, x))(outputs)  # (h*N, T_q, T_k)

    # Causality = Future blinding
    if causality:
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = Lambda(lambda x: tf.where(tf.equal(masks, 0), paddings, x), name=prefix + 'Causality')(
            outputs)  # (h*N, T_q, T_k)

    # Activation
    outputs = tf.keras.layers.Activation('softmax', name=prefix + 'weight_softmax')(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs = Lambda(lambda x: x * query_masks, name=prefix + 'q_masking')(outputs)  # broadcasting. (N, T_q, C)

    # Dropouts
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)

    # Weighted sum
    outputs = Lambda(lambda x: tf.matmul(x[0], x[1]))([outputs, V_])

    # Restore shape
    outputs = Lambda(lambda x: tf.concat(tf.split(x, num_heads, axis=0), axis=2),
                     output_shape=(-1, T, C), name=prefix + 'restore')(outputs)  # (N, T_q, C)

    # Residual connection
    outputs = tf.keras.layers.Add()([outputs, queries])

    # Normalize
    outputs = LayerNormalization(name=prefix + 'layer_norm')(outputs)
    return outputs


def positional_encoding(inputs, scale=True):
    print(type(inputs))
    N, T, num_units = inputs.get_shape().as_list()
    # First part of the PE function: sin and cos argument
    position_enc = np.array([[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    pos = np.array([position_enc for _ in range(N)])

    if scale:
        pos = pos * num_units ** 0.5

    pos = tf.convert_to_tensor(pos)
    outputs = inputs + pos
    return outputs


def make_pos(batch_size, sequence_length, num_units, scale=True):
    N, T = batch_size, sequence_length
    # First part of the PE function: sin and cos argument
    position_enc = np.array([[pos / np.power(10000, 2. * i / num_units) for i in range(num_units)] for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    outputs = np.array([position_enc for _ in range(N)])

    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs


def feed_forward(inputs, num_units=[2048, 512]):
    # Inner layer
    outputs = tf.layers.conv1d(inputs, filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)
    # Readout layer
    outputs = tf.layers.conv1d(outputs, filters=num_units[1], kernel_size=1, activation=None, use_bias=True)
    # Residual connection
    outputs = tf.keras.layers.add([outputs, inputs])
    # Normalize
    outputs = LayerNormalization()(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)
