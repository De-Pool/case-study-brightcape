import csv
import json

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
    file = open('./data/' + name)
    dialect = csv.Sniffer().sniff(file.read(4048))
    file.seek(0)
    reader = csv.reader(file, dialect)
    data = []

    for row in reader:
        row = [x for x in row if x != ""]
        data.append(row)

    matrix = np.asarray(data, dtype=np.float)
    return matrix
