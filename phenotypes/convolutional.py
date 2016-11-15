# -*- coding: utf-8 -*-
#import keras
import os, glob, logging
import numpy as np

from Bio import SeqIO
from keras.layers import Convolution1D
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

#from keras.utils.np_utils import to_categorical
HERE = os.path.dirname( __file__ )
DATA = os.path.join( HERE, 'data' )
log = logging.getLogger(__name__)

MAX_SEQUENCE = 512


def get_sequences(filename, exclude):
    """Return the sequences from a FASTA file."""
    # ToDo: Validate sequence only contains 20 standard amino acids
    sequences = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if record.id not in exclude:
                # TODO: where len(seq) > 512 in the 
                # positive set, produce
                # each 512-byte sequence in the file 
                # so that 
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

def create_ord_seq(aa_seq):
    ord_seq = [ord(char) for char in aa_seq]
    while len(ord_seq) < MAX_SEQUENCE:
        ord_seq.append(0)
    return ord_seq


def create_seq_array():
    oe_ids = get_ids(os.path.join(DATA, 'overexpression_all.fasta'))

    bg_aa_seqs = get_sequences(os.path.join(DATA, 'orf_trans.fasta'), oe_ids)
    bg_seq_array = seq_array_cats(bg_aa_seqs, 0)

    oe_aa_seqs = get_sequences(os.path.join(DATA, 'overexpression_all.fasta'), [])
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

def create_model( ):
    """Create the network model"""
    model = Sequential()
    model.add(Embedding(
        256,
        64,
        input_length=MAX_SEQUENCE, dropout=0.5
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        32, 10, border_mode='same', input_shape=(128, 64)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        32, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        16, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        16, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(Convolution1D(
        16, 3, border_mode='same', input_shape=(10,10)
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        64,
        dropout_W=.2,
        dropout_U=.2,
    ))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['binary_accuracy']
    )
    return model


def run_convo(train, test,resume=False):
    model = create_model()
    print(model.summary())
    if resume:
        weights = sorted(glob.glob(
            os.path.join( DATA, 'weights-*.hdf5' ),
        ), key=lambda x: os.stat(x).st_mtime)
        if weights:
            print( "loading weights from %s",weights[-1])
            model.load_weights(weights[-1])
    sequences = train[:, 1:]
    categories = train[:, 0]

    sequences_test = test[:, 1:]
    categories_test = test[:, 0]
    
#    early_stopping = keras.callbacks.EarlyStopping(
#        monitor='accuracy', patience=0, verbose=1, mode='auto'
#    )
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(DATA,"weights-{epoch:03d}.hdf5"), 
        verbose=1, 
    )

    model.fit(
        sequences, 
        categories,
        #to_categorical(categories.astype(bool)),
        nb_epoch=5, 
        batch_size=16,
        callbacks=[
            #early_stopping,
            checkpointer,
        ],
    )
    # Final evaluation of the model
    scores = model.evaluate(
        sequences_test, 
        categories_test,
        #to_categorical(categories_test.astype(bool)),
        verbose=1
    )
    print("Accuracy: %.2f%%" % (scores[1]*100))


def main():
    options = get_options().parse_args()
    train, test = create_seq_array()
    run_convo(train, test, resume = options.resume)

def get_options():
    import argparse
    parser = argparse.ArgumentParser( description='Run the phenotype search' )
    parser.add_argument(
        '-r','--resume',
        help = 'Resume processing using the last-saved epoch (requires an un-changed model)',
        default = False,
        action = 'store_true',
    )
    return parser

if __name__ == '__main__':
    logging.basicConfig( level=logging.INFO )
    main()
