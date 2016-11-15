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
    # crudely boost the positive signals by over-sampling
    # should investigate using a proper oversampling method..
    oe_aa_seqs = np.repeat( oe_aa_seqs, int(len(bg_aa_seqs)/len(oe_aa_seqs)))
    oe_seq_array = seq_array_cats(oe_aa_seqs, 1)

    bg_train_index = int(0.8 * len(bg_seq_array))
    oe_train_index = int(0.8 * len(oe_seq_array))

    oe_test_total = int(len(oe_seq_array) - oe_train_index)
    
    train = np.concatenate((
        bg_seq_array[:bg_train_index], 
        oe_seq_array[:oe_train_index]), 
        axis=0
    )
    test = np.concatenate((
        bg_seq_array[bg_train_index:bg_train_index+oe_test_total],
        oe_seq_array[oe_train_index:]), 
        axis=0
    )
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test


def run_convo(train, test):
    model = Sequential()
    model.add(Embedding(
        256,
        64,
        input_length=MAX_SEQUENCE, dropout=0.5
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        10, 10, border_mode='same', input_shape=(128, 64)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        10, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        10, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        10, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
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

    model.fit(sequences, categories, nb_epoch=10, batch_size=32)
    # Final evaluation of the model
    scores = model.evaluate(sequences_test, categories_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def main():
    train, test = create_seq_array()
    run_convo(train, test)



if __name__ == '__main__':
    main()
