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
"""

def get_sequences(filename):
    """Return the sequences from a FASTA file."""
    # ToDo: Validate sequence only contains 20 standard amino acids
    sequences = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            sequences.append(record.seq[:512])
    return sequences


def get_ids(filename):
    ids = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            ids.append(record.id)
    return ids


def get_total_records(filenames):
    """Return the legnths of sequences in a FASTA file."""
    count = 0
    for file in filenames:
        with open(os.path.join('data', file), 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                count += 1
    return count


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
    total_sequences = get_total_records(['orf_trans.fasta'])
    oe_ids = get_ids('data/overexpression_all.fasta')
    ord_sequences = np.zeros((total_sequences, 513), dtype=float)

    count = 0

    aa_seqs = get_sequences(os.path.join('data', 'orf_trans.fasta'))
    for aa_seq in aa_seqs:
        ord_seq = create_ord_seq(aa_seq)
        assert len(ord_seq) == MAX_SEQUENCE
        ord_sequences[count][0] = 0
        ord_sequences[count][1:] = ord_seq
        count += 1

    count = 0
    a_seqs = get_sequences(os.path.join('data', 'overexpression_all.fasta'))
    for aa_seq in a_seqs:
        ord_seq = create_ord_seq(aa_seq)
        assert len(ord_seq) == MAX_SEQUENCE
        ord_sequences[count][0] = 1
        ord_sequences[count][1:] = ord_seq
        count += 1

    return ord_sequences


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
    np.random.shuffle(seq_array)
    l = len(seq_array)
    train_index = int(0.8 * l)
    train = seq_array[:train_index]
    test = seq_array[train_index:]
    run_convo(train, test)



if __name__ == '__main__':
    main()
