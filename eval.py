# -*- coding: utf-8 -*-
from __future__ import print_function
from transformer import TransformerModel

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab


def evaluate():
    # Load model
    weight_path = 'model/08311708_epoch_0_train_loss_5.3375.h5'

    # Load data
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    model = TransformerModel(in_vocab_len=len(idx2de), out_vocab_len=len(idx2en), max_len=hp.maxlen)
    model.load_model(weight_path)

    for i in range(len(X) // hp.batch_size):
        x = X[i*hp.batch_size: (i+1)*hp.batch_size]
        sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
        targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]

        preds = model.translate(x, idx2en)

        for source, target, pred in zip(sources, targets, preds):
            print('source: {}, expected: {}, pred: {}'.format(source, target, pred))


if __name__ == '__main__':
    evaluate()
    print("Done")
