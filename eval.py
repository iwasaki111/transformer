# -*- coding: utf-8 -*-
from __future__ import print_function
from transformer import TransformerModel

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab, load_train_data


def evaluate():
    # Load model
    weight_path = 'model/09031344_epoch_4_train_loss_3.7933.h5'

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
            print('source:', source)
            print('expected:', target)
            print('pred:', pred)
            print()


def evaluate_train():
    # Load model
    weight_path = 'model/09031925_epoch_0_train_loss_5.9855.h5'

    # Load data
    Sources, Targets = load_train_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    batch_size = 5

    model = TransformerModel(in_vocab_len=len(idx2de), out_vocab_len=len(idx2en), max_len=hp.maxlen)
    model.load_model(weight_path)

    for i in range(5 // batch_size):
        x = Sources[i*batch_size: (i+1)*batch_size]
        sources = Sources[i*batch_size: (i+1)*batch_size]
        targets = Targets[i*batch_size: (i+1)*batch_size]

        preds = model.translate_with_ans(sources, targets, idx2en)
        # preds = model.translate(x, idx2en)

        for source, target, pred in zip(sources, targets, preds):
            print('source:', ' '.join(idx2de[idx] for idx in source))
            print('expected:', ' '.join(idx2en[idx] for idx in target))
            print('pred:', pred)
            print()


if __name__ == '__main__':
    evaluate_train()
    # evaluate()
    print("Done")
