import numpy as np
import implicit
from scipy import sparse

import basic_collaborative_filtering as bcf


def simple(model):
    s = 0
    for i in range(len(model.test_data)):
        s += model.predict_rating(i, int(model.test_data[i]))
    return s / len(model.test_data)


def hit_rate(model, r):
    if isinstance(model, implicit.approximate_als.AlternatingLeastSquares):
        # if instance of ALS, r = [train_matrix, test_data, r]
        matrix = sparse.csr_matrix(r['train_matrix'])

        # Fit model with transposed matrix, since we need an (item, user) matrix
        model.fit(matrix.T)
        #
        # all recommendations
        s = 0
        for i in range(len(r['test_data'])):
            if int(r['test_data'][i]) in dict(model.recommend(i, matrix, N=r['r'])):
                s += 1
        return s / len(r['test_data'])
    elif model.split_method == 'temporal':
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


def hit_rate_adam(model, r):
    if isinstance(model, implicit.approximate_als.AlternatingLeastSquares):
        # if instance of ALS, r = [train_matrix, test_data, r]
        matrix = sparse.csr_matrix(r['train_matrix'])

        # Fit model with transposed matrix, since we need an (item, user) matrix
        model.fit(matrix.T)
        #
        # all recommendations
        s = 0
        for i in range(len(r['test_data'])):
            if int(r['test_data'][i]) in dict(model.recommend(i, matrix, N=r['r'])):
                s += 1
        return s / len(r['test_data'])
    elif model.split_method == 'temporal':
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

        return recommendations