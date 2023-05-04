import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import os
from enum import Enum, auto
import multiprocessing as mp
import pickle
import time
from stollen_prob_search import StolenProbabilitySearch, ExactAlgorithms, ApproxAlgorithms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(15.7, 12.27)})


def get_time_elapsed(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2}"


def linearly_separable(X):
    n, d = X.shape
    sp_search = StolenProbabilitySearch(X)
    results = sp_search.find_bounded_classes(class_list=tuple(range(X.shape[0])),
                                                exact_algorithm=ExactAlgorithms.default(),
                                                approx_algorithm=ApproxAlgorithms.default(),
                                                lb=-100,
                                                ub=100,
                                                patience=100, 
                                                desc=f" n: {n:>3}, d: {d:>3}")
    bounded = [r['index'] for r in results if r['is_bounded']]
    return len(bounded) > 0

def get_X(d, n, init, norm):
    init_func = init.get_initialization_func()
    norm_func = norm.get_normalization_func()
    X = torch.rand(n, d)
    X = init_func(X)
    X = norm_func(X)
    return X

def run_exp(args):
    d, n, init, norm, runs = args
    start = time.time()
    acc = 0.
    run = 0
    tries = runs
    while run < runs:
        try:
            X = get_X(d, n, init, norm)
            acc += linearly_separable(X, multiprocess=False)
        except Exception as e:
            print(e)
            if tries == 0:
                print(f"Tries exhausted for n={n} d={d} run={run}")
            tries -= 1
            continue
        tries = runs
        run += 1

    acc /= runs
    elapsed = get_time_elapsed(start)
    return n, d, acc, elapsed



class INITIALIZATION(Enum):
    UNIFORM = auto()
    NORMAL = auto()
    XAVIER_UNIFORM = auto()
    XAVIER_NORMAL = auto()

    @staticmethod
    def from_string(s):
        try:
            return INITIALIZATION[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        if self is INITIALIZATION.UNIFORM:
            return "UNIFORM"
        elif self is INITIALIZATION.NORMAL:
            return "NORMAL"
        elif self is INITIALIZATION.XAVIER_UNIFORM:
            return "XAVIER_UNIFORM"
        elif self is INITIALIZATION.XAVIER_NORMAL:
            return "XAVIER_NORMAL"
        return "NA"

    def get_initialization_func(self):
        if self is INITIALIZATION.UNIFORM:
            return nn.init.uniform_
        elif self is INITIALIZATION.NORMAL:
            return nn.init.normal_
        elif self is INITIALIZATION.XAVIER_UNIFORM:
            return nn.init.xavier_uniform_
        elif self is INITIALIZATION.XAVIER_NORMAL:
            return nn.init.xavier_normal_


def layer_norm_init(X):
    if len(X.shape) > 2:
        b, n, d = X.shape
    else:
        n, d = X.shape
    if d < 2:
        raise Exception("d must be bigger than 1")
    if d > 2:
        X = (X - X.mean(dim=-1, keepdim=True)) / X.std(dim=-1, keepdim=True, unbiased=False)
    elif d == 2:
        # raise NotImplementedError
        x = torch.rand(n)
        y_square = (1 - x ** 2) ** 0.5
        sign = (torch.rand(n) < 0.5) * 2 - 1
        y = y_square * sign
        X = torch.stack([x, y], dim=1)
        X *= d ** 0.5
    return X


class NORMALIZATION(Enum):
    LAYER_NORM = auto()
    NONE = auto()

    @staticmethod
    def from_string(s):
        try:
            return NORMALIZATION[s]
        except KeyError:
            raise ValueError()

    def __str__(self):
        if self is NORMALIZATION.LAYER_NORM:
            return "LAYER_NORM"
        elif self is NORMALIZATION.NONE:
            return "NONE"
        return "NA"

    def get_normalization_func(self):
        if self is NORMALIZATION.LAYER_NORM:
            return layer_norm_init
        elif self is NORMALIZATION.NONE:
            return lambda x: x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_d', default=3, type=int,
                        help='Min input dimension')
    parser.add_argument('--max_d', default=15, type=int,
                        help='Max input dimension')
    parser.add_argument('--min_n', default=3, type=int,
                        help='Min input dimension')
    parser.add_argument('--max_n', default=60, type=int,
                        help='Max input vectors')
    parser.add_argument("--initialization", dest="initialization", default=INITIALIZATION.NORMAL,
                        type=INITIALIZATION.from_string, choices=list(INITIALIZATION), required=False,
                        help='Initialization type')
    parser.add_argument("--normalization", dest="normalization", default=NORMALIZATION.NONE,
                        type=NORMALIZATION.from_string, choices=list(NORMALIZATION), required=False,
                        help='Initialization type')
    parser.add_argument('--runs', default=100, type=int,
                        help='Number of runs')
    parser.add_argument('--out', default="unselectable_rand_exp_out", type=str,
                        help='Out dir')
    parser.add_argument('--seed', default=123, type=int,
                        help='Random seed (default: 123)')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    start = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    min_d = args.min_d
    max_d = args.max_d
    min_n = args.min_n
    max_n = args.max_n
    acc_mat = np.empty((max_n - min_n + 1, max_d - min_d + 1))

    exp_args = [(d, n, args.initialization, args.normalization, args.runs) for d in range(min_d, max_d + 1) for n in range(min_n, max_n + 1)]
    total = len(exp_args)
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        for i, (n, d, acc, elapsed) in enumerate(pool.imap(run_exp, exp_args)):
            print(f"n: {n:>3}, d: {d:>3}: acc: {acc:>5.2f} elapsed: {elapsed} ({(i + 1)}/{total})")
            acc_mat[n - min_n, d - min_d] = acc

    print("Acc")
    print(acc_mat)

    print(f"\nTotal time: {get_time_elapsed(start)}")

    filename = f"unselectable_rand_n=[{args.min_n},{args.max_n}]_d=[{args.min_d},{args.max_d}]_{str(args.initialization)}_{str(args.normalization)}"
    with open(os.path.join(args.out, f"{filename}.pickle"), "wb") as f:
        pickle.dump({'acc_mat': acc_mat,
                     'args': args}, f, protocol=pickle.HIGHEST_PROTOCOL)

    mask = np.isnan(acc_mat)

    yticklabels = list(range(min_n, max_n + 1))
    xticklabels = list(range(min_d, max_d + 1))
    ax = sns.heatmap(acc_mat,
                     vmin=0,
                     vmax=1,
                     cmap=sns.color_palette("rocket_r", as_cmap=True),
                     xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_xlabel("d")
    ax.set_ylabel("n")
    plt.savefig(os.path.join(args.out, f"{filename}.pdf"))

