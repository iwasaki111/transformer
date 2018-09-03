# -*- coding: utf-8 -*-
import os
import cv2
import pytz
from datetime import datetime
import numpy as np
import tensorflow as tf

from modules import multi_head_attention, feed_forward, label_smoothing, positional_encoding, make_pos, Embedding


class TrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, predict_model, output_dir):
        self.predict_model = predict_model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        time_stamp = datetime.strftime(datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
        save_model_path = '{}_epoch_{}_train_loss_{:.4f}.h5'.format(time_stamp, epoch, logs['loss'])
        save_model_path = os.path.join(self.output_dir, save_model_path)
        self.predict_model.save_weights(save_model_path)


class TransformerModel(object):
    def __init__(self, in_vocab_len, out_vocab_len, max_len, units=512):
        self.in_vocab_len = in_vocab_len
        self.out_vocab_len = out_vocab_len
        self.max_len = max_len
        self.units = units
        self._build(rnn_size=units)

    def _build(self, num_blocks=6, num_heads=8, rnn_size=512):
        x_input = tf.keras.layers.Input((self.max_len,), dtype='int32')
        y_input = tf.keras.layers.Input((self.max_len,), dtype='int32')
        decoder_input = tf.keras.layers.Lambda(lambda x: tf.concat((tf.ones_like(x[:, :1]) * 2, x[:, :-1]), -1))(y_input)

        pos = tf.keras.layers.Input((None, rnn_size))

        # Encoder
        enc = Embedding(self.in_vocab_len, rnn_size)(x_input)

        # positional encode
        enc = tf.keras.layers.add([enc, pos])
        enc = tf.keras.layers.Dropout(0.1)(enc)

        for i in range(num_blocks):
            enc = multi_head_attention(enc, enc, 'enc_block_{}'.format(i + 1), num_units=rnn_size, num_heads=num_heads,
                                       causality=False)
            enc = feed_forward(inputs=enc, num_units=[1024, 512])

        # Decoder
        dec = Embedding(self.out_vocab_len, rnn_size)(decoder_input)

        # positional encode
        dec = tf.keras.layers.add([dec, pos])

        dec = tf.keras.layers.Dropout(0.1)(dec)

        for i in range(num_blocks):
            dec = multi_head_attention(dec, dec, 'dec_block_{}_1'.format(i + 1), num_units=rnn_size,
                                       num_heads=num_heads, causality=True)
            dec = multi_head_attention(dec, enc, 'dec_block_{}_2'.format(i + 1), num_units=rnn_size,
                                       num_heads=num_heads, causality=False)
            dec = feed_forward(inputs=dec, num_units=[1024, 512])

        logits = tf.keras.layers.Dense(self.out_vocab_len)(dec)
        loss = tf.keras.layers.Lambda(self.loss_function, output_shape=(1,), name='loss')([logits, y_input])

        self.predict_model = tf.keras.Model(inputs=[x_input, y_input, pos], outputs=logits)
        self.train_model = tf.keras.Model(inputs=[x_input, y_input, pos], outputs=loss)

    def load_model(self, load_weight_path):
        self.predict_model.load_weights(load_weight_path)

    def translate(self, x, idx2en, batch_size=4):
        texts = []
        pos = make_pos(len(x), self.max_len, self.units)
        preds = np.zeros(x.shape)
        for i in range(self.max_len):
            # inputs = np.array()
            _preds = self.predict_model.predict([x, preds, pos], batch_size=batch_size)
            _preds = np.argmax(_preds, axis=-1)
            preds[:, i] = _preds[:, i]
            print(' '.join(idx2en[idx] for idx in _preds[0, :i]))

        for pred in preds:
            texts.append(' '.join(idx2en[idx] for idx in pred).split('</S>')[0].strip())
        return texts

    def translate_with_ans(self, x, y, idx2en, batch_size=4):
        texts = []
        pos = make_pos(len(x), self.max_len, self.units)
        preds = self.predict_model.predict([x, y, pos], batch_size=batch_size)
        preds = np.argmax(preds, axis=-1)
        for pred in preds:
            texts.append(' '.join(idx2en[idx] for idx in pred).split('</S>')[0].strip())
        return texts

    def loss_function(self, inputs):
        logits, label = inputs
        istarget = tf.to_float(tf.not_equal(label, 0))
        y_smoothed = label_smoothing(tf.one_hot(label, depth=self.out_vocab_len))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)
        mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))
        return mean_loss

    def create_optimizer(self, learning_rate, optimizer):
        if optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        elif optimizer == 'ADAGRAD':
            opt = tf.keras.optimizers.Adagrad(lr=learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.keras.optimizers.Adadelta(lr=learning_rate)
        elif optimizer == 'ADAM':
            opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-8)
        elif optimizer == 'RMSPROP':
            opt = tf.keras.optimizers.RMSprop(lr=learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        return opt

    def training(self, x, y, epochs, batch_size, learning_rate, optimizer, output_dir='model'):
        output_dummy = np.zeros((len(x), 1))
        pos = make_pos(len(x), self.max_len, self.units)

        opt = self.create_optimizer(learning_rate, optimizer)
        train_callback = TrainCallback(self.predict_model, output_dir)

        self.train_model.compile(loss=lambda ans, pred: pred, optimizer=opt)
        self.train_model.fit(x=[x, y, pos], y=output_dummy, batch_size=batch_size, epochs=epochs,
                             callbacks=[train_callback])

    def show_model(self):
        self.predict_model.summary()
        tf.keras.utils.plot_model(self.predict_model, show_shapes=True)


def model_test():
    transformer = TransformerModel(in_vocab_len=4, out_vocab_len=8, max_len=16)
    transformer.train_model.summary()
    tf.keras.utils.plot_model(transformer.predict_model, show_shapes=True)


if __name__ == '__main__':
    model_test()
