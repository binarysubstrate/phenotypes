# -*- coding: utf-8 -*-
import os

import numpy as np
from Bio import SeqIO


def load_fasta(filename):
    """Return the legnths of sequences in a FASTA file."""
    lengths = []
    with open(filename, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            lengths.append(len(record))
    return lengths


def main():
    file_names = [
        'heritable.fasta', 'hsp104.fasta', 'hsp70.fasta', 'nc.fasta',
        'orf_trans.fasta', 'overexpression_all.fasta']

    all_lengths = []
    for file_name in file_names:
        file_lengths = load_fasta(os.path.join('data', file_name))
        all_lengths.extend(file_lengths)

    all_lengths.sort()
    print(
        "Mean of the smallest "
        "100 sequences: {}".format(np.mean(all_lengths[:5])))
    print("Mean of the largest "
          "100 sequences: {}".format(np.mean(all_lengths[-5:])))
    print("Mean of all sequences: {}".format(np.mean(all_lengths)))
    print("Number of sequences across all files: {}".format(len(all_lengths)))
    return None


if __name__ == '__main__':
    main()
