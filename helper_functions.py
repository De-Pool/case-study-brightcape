import json

import matplotlib.pyplot as plt
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


# Plot 3 lines in the same graph.
def plot3(xs, ys, labels, x_lbl, title, legend=True):
    fig, host = plt.subplots(figsize=(8, 5))

    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel(x_lbl)
    host.set_ylabel(labels[0])
    par1.set_ylabel(labels[1])
    par2.set_ylabel(labels[2])

    line1, = host.plot(xs, ys[0], color='red', label=labels[0])
    line2, = par1.plot(xs, ys[1], color='green', label=labels[1])
    line3, = par2.plot(xs, ys[2], color='blue', label=labels[2])

    if legend:
        host.legend(handles=[line1, line2, line3], loc='lower right')

    par2.spines['right'].set_position(('outward', 60))
    host.yaxis.label.set_color(line1.get_color())
    par1.yaxis.label.set_color(line2.get_color())
    par2.yaxis.label.set_color(line3.get_color())

    plt.title(title)
    fig.tight_layout()
    plt.show()


# Plot 2 lines in the same graph.
def plot2(xs, ys, labels, x_lbl, y_lbls, title, legend=True):
    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    ax.plot(xs[0], ys[0], label=labels[0], color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbls[0])

    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()
    ax2.plot(xs[1], ys[1], label=labels[1], color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel(y_lbls[1])
    plt.title(title)
    if legend:
        fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.14))
        plt.tight_layout()
    plt.show()


# Plot l lines with the same x coordinates.
def plot(x, y, label, x_lbl, y_lbl, title, legend=True):
    for i in range(len(y)):
        plt.plot(x, y[i], label=label[i])
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    if legend:
        plt.legend()
    plt.show()


def create_x_y(all_params, num_params, parameter):
    x = []
    y = []
    for i in range(len(all_params)):
        x.append(all_params[i][parameter])
        y.append([all_params[i][num_params][0], all_params[i][num_params][1], all_params[i][num_params][2]])
    return np.array(x), np.array(y).T
