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


def save_matrix(matrix, name):
    np.savetxt('./data/' + name, matrix, delimiter=',')


def read_matrix(name):
    # Don't use np.gentxt, since it is not optimized.
    chunks = pd.read_csv('./data/' + name, chunksize=1000000, header=None)
    df = pd.concat(chunks)

    return df.values
