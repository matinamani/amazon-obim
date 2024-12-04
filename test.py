import numpy as np
import pandas as pd


corpus = [
    'where python is used',
    'what is python used in',
    'why python is best',
    'what companies use python'
]

_words = ' '.join(corpus).split()
words = list(set(_words))
n = len(words)
c_matrix = np.zeros((n, n), dtype='int')

for context in corpus:
    context = context.split()
    index = [words.index(item) for item in context]
    for i in range(len(context)):
        for j in range(i, len(context)):
            if not i == j:
                c_matrix[index[j]][index[i]] += 1

pd.DataFrame(c_matrix)
