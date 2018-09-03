# -*- coding: utf-8 -*-
from __future__ import print_function

from hyperparams import Hyperparams as hp
from data_load import load_train_data, load_de_vocab, load_en_vocab
from transformer import TransformerModel

if __name__ == '__main__':
    # Load vocabulary    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Construct graph
    model = TransformerModel(in_vocab_len=len(idx2de), out_vocab_len=len(idx2en), max_len=hp.maxlen)
    X, Y = load_train_data()
    # X, Y = X[:100], Y[:100]  # test
    print(X.shape)
    print(Y.shape)
    model.show_model()
    model.training(x=X, y=Y, epochs=hp.num_epochs, batch_size=hp.batch_size, learning_rate=hp.lr, optimizer=hp.optimizer)

    print("Done")
