import json

import pandas as pd

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np


def save_dict(dictionary, name):
    json_file = open('./data/' + name, 'w')
    json.dump(dictionary, json_file)


def read_dict(name):
    return json.load(open('./data/' + name))


def save_matrix(matrix, name):
    np.savetxt('./data/' + name, matrix, delimiter=',')


def read_matrix(name):
    # Don't use np.gentxt, since it is not optimized.
    chunks = pd.read_csv('./data/' + name, chunksize=1000000, header=None)
    df = pd.concat(chunks)

    return df.values


def concat(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return [a, b]
    elif isinstance(a, int) and isinstance(b, list):
        return [a, *b]
    elif isinstance(a, list) and isinstance(b, int):
        return [*a, b]
    elif isinstance(a, np.int32) and isinstance(b, list):
        return [a.tolist(), *b]
    elif isinstance(a, np.int32) and isinstance(b, int):
        return [a.tolist(), b]
    elif isinstance(a, np.int32) and isinstance(b, np.int32):
        return [a.tolist(), b.tolist()]
    elif isinstance(a, list) and isinstance(b, np.int32):
        return [*a, b.tolist()]
    elif isinstance(a, int) and isinstance(b, np.int32):
        return [a, b.tolist()]
    elif isinstance(a, list) and isinstance(b, list):
        return [*a, *b]
