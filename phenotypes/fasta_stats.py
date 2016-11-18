# -*- coding: utf-8 -*-
import os

import numpy as np
from Bio import SeqIO


def load_fasta(filename):
    """Return the lengths of sequences in a FASTA file."""
    lengths = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            lengths.append(len(record))
    return lengths


def print_seq_lengths(fasta_files):
    all_lengths = []
    for file_name in fasta_files:
        file_lengths = load_fasta(os.path.join('data', file_name))
        all_lengths.extend(file_lengths)

    all_lengths.sort()
    print("Mean of the smallest "
          "100 sequences: {}".format(np.mean(all_lengths[:100])))
    print("Mean of the largest "
          "100 sequences: {}".format(np.mean(all_lengths[-100:])))
    print("Mean of all sequences: {}".format(np.mean(all_lengths)))
    print("Maximum sequence length: {}".format(all_lengths[-1]))
    print("Number of sequences across all files: {}".format(len(all_lengths)))
    return


def print_possible_chars(fasta_files):
    all_chars = set()
    for file_name in fasta_files:
        with open(os.path.join('data', file_name), 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                for char in record.seq:
                    all_chars.update(char)

    print("Set of all possible characters:\n{}".format(all_chars))
    print("Number of all possible characters: {}".format(len(all_chars)))

    return None


def localization_stats(fasta_files):
    print("Localization stats:")
    total = 0
    for fasta in fasta_files:
        total += len(load_fasta(os.path.join('data/', fasta)))
        print(len(load_fasta(os.path.join('data/', fasta))))
    print(1411 / total)


def main():
    fasta_set1 = [
        'heritable.fasta', 'hsp104.fasta', 'hsp70.fasta', 'nc.fasta',
        'orf_trans.fasta', 'overexpression_all.fasta'
    ]

    fasta_set2 = [
        'chloroplast.fasta', 'cytoplasmic.fasta', 'ER.fasta',
        'extracellular.fasta', 'Golgi.fasta', 'nuclear.fasta',
        'peroxisomal.fasta', 'plasma_membrane.fasta', 'vacuolar.fasta'
    ]

    localization_stats(fasta_set2)
    print("\n{}\n".format("-"*30))
    print_possible_chars(fasta_set2)
    print("\n{}\n".format("-"*30))
    print_seq_lengths(fasta_set2)


if __name__ == '__main__':
    main()
