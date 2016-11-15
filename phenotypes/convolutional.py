# -*- coding: utf-8 -*-
import keras
import os
import numpy as np

from Bio import SeqIO
from keras.layers import Convolution1D
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding

MAX_SEQUENCE = 512

"""
ToDo:
Remove positives from negative set
Try a set with all the separate categories
Find a way to include the entire sequence (break up into chunks?)
Do a 50/50 set - especially make a test set that is 50/50
Look at 1 neuron vs. 20
set numbers to be a smaller subset
"""

def get_sequences(filename, exclude):
    """Return the sequences from a FASTA file."""
    # ToDo: Validate sequence only contains 20 standard amino acids
    sequences = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if record.id not in exclude:
                sequences.append(record.seq[:512])
    return sequences


def seq_array_cats(sequences, cat):
    ord_sequences = np.zeros((len(sequences), 513), dtype=float)
    count = 0
    for aa_seq in sequences:
        ord_seq = create_ord_seq(aa_seq)
        assert len(ord_seq) == MAX_SEQUENCE
        ord_sequences[count][0] = cat
        ord_sequences[count][1:] = ord_seq
        count += 1
    return ord_sequences


def get_ids(filename):
    ids = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            ids.append(record.id)
    return ids


def convo():
    nb_filter = 32  # Size of the kernel
    filter_length = 3  # Micro patterns

    keras.layers.convolutional.Convolution1D(
        nb_filter, filter_length, init='uniform', activation='linear',
        weights=None, border_mode='valid', subsample_length=1,
        W_regularizer=None, b_regularizer=None, activity_regularizer=None,
        W_constraint=None, b_constraint=None, bias=True, input_dim=None,
        input_length=None)

    # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
    # with 64 output filters
    model = Sequential()
    model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
    # now model.output_shape == (None, 10, 64)

    # add a new conv1d on top
    model.add(Convolution1D(32, 3, border_mode='same'))
    # now model.output_shape == (None, 10, 32)
    return None


def create_ord_seq(aa_seq):
    ord_seq = [ord(char) for char in aa_seq]
    while len(ord_seq) < MAX_SEQUENCE:
        ord_seq.append(0)
    return ord_seq


def create_seq_array():
    oe_ids = get_ids(os.path.join('data', 'overexpression_all.fasta'))

    bg_aa_seqs = get_sequences(os.path.join('data', 'orf_trans.fasta'), oe_ids)
    bg_seq_array = seq_array_cats(bg_aa_seqs, 0)

    oe_aa_seqs = get_sequences(os.path.join('data', 'overexpression_all.fasta'), [])
    oe_seq_array = seq_array_cats(oe_aa_seqs, 0)

    bg_train_index = int(0.8 * len(bg_seq_array))
    oe_train_index = int(0.8 * len(oe_seq_array))

    array_size = bg_train_index + oe_train_index

    ord_sequences = np.zeros((array_size, MAX_SEQUENCE + 1), dtype=float)

    oe_test_total = int(len(oe_seq_array) - oe_train_index)
    train = np.append(bg_seq_array[:bg_train_index], oe_seq_array[:oe_train_index])
    test = np.append(bg_seq_array[bg_train_index:bg_train_index+oe_test_total],
                     oe_seq_array[oe_train_index:])
    np.random.shuffle(train)
    np.random.shuffle(test)

    return train, test


def run_convo(train, test):
    ord_sequences = create_seq_array()
    # TODO: Batch sequences to convo network.
    model = Sequential()
    model.add(Embedding(
        256,
        64,
        input_length=MAX_SEQUENCE, dropout=0.5
    ))
    model.add(Convolution1D(
        64, 3, border_mode='same', input_shape=(128, 64)
    ))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())
    sequences = train[:, 1:]
    categories = train[:, 0]

    sequences_test = test[:, 1:]
    categories_test = test[:, 0]


    model.fit(sequences, categories, nb_epoch=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(sequences_test, categories_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def main():
    seq_array = create_seq_array()
    print(seq_array)

    #run_convo(train, test)



if __name__ == '__main__':
    main()
