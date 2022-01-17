import numpy as np

import basic_collaborative_filtering as bcf


def simple(model):
    s = 0
    for i in range(len(model.test_data)):
        s += model.predict_rating(i, int(model.test_data[i]))
    return s / len(model.test_data)


def hit_rate(model, r):
    if model.split_method == 'temporal':
        s = 0
        total = 0
        recommendations = bcf.predict_recommendation(model.ratings_matrix, model.n, r)
        for i in range(len(model.test_data)):
            s += len(np.intersect1d(np.array(model.test_data[i]), recommendations[i].astype(int)))
            total += len(model.test_data[i])
        return s / total
    else:
        s = 0
        recommendations = bcf.predict_recommendation(model.ratings_matrix, model.n, r)
        for i in range(len(model.test_data)):
            if int(model.test_data[i]) in recommendations[i].astype(int):
                s += 1

        return s / len(model.test_data)
