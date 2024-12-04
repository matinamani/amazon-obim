import numpy as np
import pandas as pd
from reviewCleaner import cleaner
from statusTools import *
from pathlib import Path
from collections import defaultdict


def make_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class Normalizer:
    def __init__(self, brand, aspects=['phone', 'camera'], stats=True, cleanup=True, window_size=1):
        self.brand = brand
        self.aspects = aspects
        self.stats = stats
        self.cleanup = cleanup
        self.window_size = window_size
        self.corpus = pd.read_csv(f'./assets/reviews/{brand}.csv')[
            'Review'].map(lambda r: cleaner(r) if cleanup else r.lower().strip())
        self.words = ' '.join(self.corpus).split()
        self.unqWords = list(set(self.words))  # words no duplications
        self.n = len(self.unqWords)

    def normalize(self):
        if self.stats:
            status(
                f'------Normalizing {self.brand} with Window Size of {self.window_size}------')

        self.calc_coMatrix()
        self.calc_degree()
        self.calc_wc()
        self.calc_v()
        self.calc_nc()
        self.calc_strength()
        self.calc_mu()
        self.calc_uniqueness()

    def calc_coMatrix(self):
        # code from https://stackoverflow.com/a/58725727/12278890
        if self.stats:
            status(f'Calculating Co-Occurrence Matrix for {self.brand}...')

        d = defaultdict(int)
        vocab = set()
        for text in self.corpus:
            # preprocessing (use tokenizer instead)
            text = text.split()

            # iterate over sentences
            for i in range(len(text)):
                token = text[i]
                vocab.add(token)
                next_token = text[i + 1: i + 1 + self.window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]))
                    d[key] += 1

        # formulate the dictionary into dataframe
        vocab = sorted(vocab)  # sort vocab
        self.CoMatrix = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                                     index=vocab,
                                     columns=vocab)

        for key, value in d.items():
            self.CoMatrix.at[key[0], key[1]] = value
            self.CoMatrix.at[key[1], key[0]] = value

    def calc_degree(self):
        if self.stats:
            status(f'Calculating Words Degree for {self.brand}...')

        self.deg = self.CoMatrix.apply(lambda column: len(column[column > 0]))

    def calc_wc(self):
        if self.stats:
            status(f'Calculating Word Count for {self.brand}...')

        dict = {}
        for word in self.words:
            dict.update({word: dict.get(word, 0) + 1})

        self.wc = pd.Series(dict)

    def calc_v(self):
        if self.stats:
            status(f'Calculating Number of Unique Words for {self.brand}...')

        self.v = len(self.wc[self.wc == 1])
        self.otherwise = 1 / self.v

    def calc_nc(self):
        if self.stats:
            status(f'Calculating NC for {self.brand}...')

        self.nc = self.CoMatrix.apply(
            lambda row: self.CoMatrix[row.name]/self.wc[row.name],
            axis=1).replace(1, self.otherwise)

    def calc_strength(self):
        if self.stats:
            status(f'Calculating Strength for {self.brand}...')

        self.strength = self.nc.apply(lambda row: row.sum() / self.deg[row.name],
                                      axis=1).sort_values(ascending=False)

    def calc_mu(self):
        if self.stats:
            status(f'Calculating Mu for {self.brand}...')

        self.mu = pd.DataFrame(index=self.aspects, columns=self.aspects)

        for i in self.aspects:
            for j in self.aspects:
                if self.deg[i] == 1 or self.deg[i] + self.deg[j] == 2:
                    self.mu[i][j] = self.nc[i][j]
                else:
                    self.mu[i][j] = self.nc[i][j] * \
                        ((self.deg[i] - 1) / (self.deg[i] + self.deg[j] - 2))

        self.nBar = self.mu.apply(
            lambda x: self.deg[x.name] / (len(self.unqWords) - 1), axis=1)

    def calc_uniqueness(self):
        if self.stats:
            status(f'Calculating Uniqueness for {self.brand}...')

        self.uniqueness = self.mu.apply(
            lambda row: self.nBar[row.name] + row.sum(), axis=1)
