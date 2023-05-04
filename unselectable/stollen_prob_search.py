import time
import numpy as np
import gurobipy as gp

from enum import Enum
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
from gurobipy import GRB

##########################################################ß#
# Adopted from https://github.com/andreasgrv/unargmaxable #
###########################################################

# Global variables used in multithreading
# This was much more efficient than using multithreading shared_memory
# https://stackoverflow.com/questions/14124588/shared-memory-in-multiprocessing
W = None
b = None

# Default bounds used if not overrided.
LB = -100
UB = 100


def is_in_bounds(p, lb, ub):
    in_bounds = True
    for coord in p:
        # Check if one coordinate is out of bounds
        if coord < lb or coord > ub:
            in_bounds = False
    return in_bounds


class ApproxAlgorithms(Enum):
    # Braid Reflect algorithm
    braid_swap = 0
    none = 2

    @classmethod
    def choices(cl):
        return [c.name for c in cl]

    @classmethod
    def default(cl):
        return cl.choices()[0]
    
    
class ExactAlgorithms(Enum):
    lp_chebyshev = 2
    none = 3

    @classmethod
    def choices(cl):
        return [c.name for c in cl]

    @classmethod
    def default(cl):
        return cl.choices()[0]


class StolenProbabilitySearch(object):
    """Algorithm to detect whether it is possible to assign all classes
    the largest probability. 
    Accepts the softmax layer parameters W and b as input."""
    def __init__(self, W, b=None):
        super(StolenProbabilitySearch, self).__init__()
        self.W = W
        self.num_classes, self.dim = self.W.shape
        self.b = b

    def find_bounded_classes(self,
                             class_list=None,
                             approx_algorithm=ApproxAlgorithms.default(),
                             exact_algorithm=ExactAlgorithms.default(),
                             lb=LB,
                             ub=UB,
                             patience=100,
                             desc=""):
        global W, b, tW, tb

        if class_list is None:
            class_list = tuple(range(self.num_classes))

        # List of classes to return
        results = []

        # We assume our vectors span the column space (dim dimension)
        # S = np.linalg.svd(self.W, compute_uv=False)
        rank = np.linalg.matrix_rank(self.W.astype(np.float64))
        assert rank == min(self.dim, self.num_classes), 'Rank=%d, dim=%d' % (rank, self.dim)
        # If we don't have at least dim + 2 weight vectors,
        # there is no way to have one weight vector be internal
        # to the convex hull of the rest.
        # E.g. in dim=2 need 4 points - 3 points form a triangle,
        # need one more point to place it inside the triangle.
        if self.num_classes < self.dim + 1:
            return results

        is_bounded = partial(class_is_bounded,
                             shape=(self.num_classes, self.dim),
                             dtype=self.W.dtype,
                             approx_algorithm=approx_algorithm,
                             exact_algorithm=exact_algorithm,
                             lb=lb,
                             ub=ub,
                            #  W=self.W,
                            #  b=self.b,
                             patience=patience)

        # Set global variables - they will be visible in threads
        W = self.W
        b = self.b

        
        with Pool(processes=mp.cpu_count() - 1) as p:
            with tqdm(total=len(class_list), desc='Checking for stolen probability' + desc, leave=False, dynamic_ncols=True, disable=False) as pbar:
                for i, result in enumerate(p.imap_unordered(is_bounded, class_list)):
                    results.append(result)
                    pbar.update()

        return list(sorted(results, key=lambda x: x['index']))


def class_is_bounded(class_idx,
                     shape,
                     dtype,
                     approx_algorithm,
                     exact_algorithm=None,
                     lb=LB,
                     ub=UB,
                    #  W=W,
                    #  b=b,
                     patience=100):

    start_time = time.time()
    result = dict()
    approx_result = dict()
    exact_result = dict()

    num_classes, dim = shape

    if lb is None:
        lb = -np.inf
    if ub is None:
        ub = np.inf

    assert (approx_algorithm is not None) or (exact_algorithm is not None)
    # NOTE: In approx method we do not include bias for time being as faster.
    # This can mean more false positives - but we can discard those with
    # exact method
    if approx_algorithm is not None:
        approx_enum = ApproxAlgorithms[approx_algorithm]
        if approx_enum == ApproxAlgorithms.braid_swap:
            approx_result = candidate_is_bounded(class_idx, W, b=b, lb=lb, ub=ub, patience=patience)
        else:
            raise ValueError('Unknown approximate algorithm: "%s"' % approx_algorithm)

    if exact_algorithm is not None:
        if approx_algorithm is None or approx_result['is_bounded']:
            exact_enum = ExactAlgorithms[exact_algorithm]
            if exact_enum == ExactAlgorithms.lp_chebyshev:
                # This takes bias term into account
                exact_result = lp_chebyshev(class_idx, W, b, lb=lb, ub=ub)
            else:
                raise ValueError('Unknown exact algorithm: "%s"'
                                 % exact_algorithm)

    result.update(**approx_result)
    result.update(**exact_result)

    result['index'] = class_idx
    # Verify Solution if we found one
    if not result['is_bounded']:
        if b is not None:
            act = W.dot(result['point']).reshape(-1, 1) + b
        else:
            act = W.dot(result['point'])
        assert np.argmax(act.ravel()) == class_idx

    end_time = time.time()
    result['time_taken'] = end_time - start_time

    return result


def lp_chebyshev(position, W, b=None, lb=LB, ub=UB):
    """Linear programme that computes maximum bounded sphere."""

    assert lb < ub
    assert lb != -np.inf
    assert ub != np.inf
    num_classes, dim = W.shape
    EPSILON = 1e-8

    # NOTE: For LP we want the halfspace defined by <= 0
    # So we subtract position from rest
    braid = W - W[position, :]
    braid = np.delete(braid, position, axis=0)

    if b is not None:
        braid_b = b - b[position]
        braid_b = np.delete(braid_b, position, axis=0)
        braid_b = braid_b.ravel()
    else:
        braid_b = np.zeros(num_classes - 1)

    c = np.zeros(dim)

    # lp = linprog(c, A_ub=braid, b_ub=braid_b, bounds=(None, None), method='highs-ipm')
    # lp = linprog(c, A_ub=braid, b_ub=braid_b, bounds=(None, None))
    # bounded = not lp.success

    # Use Gurobi
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            # m = gp.Model('m')
            m.setParam('OutputFlag', 0)
            # m.setParam('Presolve', 1)
            m.setParam('Method', 2)
            # Radius lower bound is above this
            m.setParam('FeasibilityTol', EPSILON *.1)
            m.params.threads = 1
            # NOTE: We need to specify ub and lb here otherwise there are definitely
            # unbounded regions.
            x = m.addMVar(lb=lb, ub=ub, shape=dim, name='xx')

            r = m.addMVar(lb=EPSILON, shape=1, name='r')
            # Add Chebyshev column
            cheby = np.linalg.norm(braid, axis=1, keepdims=True)

            m.addConstr(braid @ x + cheby @ r <= -braid_b, name='cc')
            m.setObjective(r, GRB.MAXIMIZE)
            m.update()

            try:
                m.optimize()
            except Exception as e:
                print(e)
            # If we find a feasible solution, class is not bounded.
            if m.status == GRB.OPTIMAL:
                is_bounded = False
                var_names = ['xx[%d]' % i for i in range(dim)]
                point = np.array([m.getVarByName(n).X for n in var_names],
                                dtype=np.float64)

                result = dict(is_bounded=is_bounded,
                            status=m.status,
                            point=point,
                            radius=getattr(m, 'objval', None))
            else:
                is_bounded = True
                result = dict(is_bounded=is_bounded,
                            status=m.status,
                            radius=getattr(m, 'objval', None))
            return result


def candidate_is_bounded(candidate_idx, W, b=None, lb=LB, ub=UB, patience=100):
    num_classes, dim = W.shape
    total_patience = patience
    # Check if the actual weight for this class is an input point that
    # makes this class be the argmax
    # We [] the first argument, so that point retains first dim (*1*, DIM)
    point = W[[candidate_idx], :]
    # transpose to (DIM, *1*)
    point = point.T
    # Obtain activation by computing matrix multiplication
    prev_argmax = get_swap_target(candidate_idx, point, W, b)

    for pat in range(patience):
        if prev_argmax == candidate_idx:
            break
        # Try swapping the elements at these two indices
        # by reflecting the point past the hyperplane of the braid arrangement
        # that swaps these two coordinates in the output space
        point = swap(candidate_idx, prev_argmax, point, W, b)

        argmax = get_swap_target(candidate_idx, point, W, b)

        prev_argmax = argmax

    is_bounded = True
    if prev_argmax == candidate_idx:
        if is_in_bounds(point, lb, ub):
            is_bounded = False

    result = dict(is_bounded=is_bounded,
                  point=point.ravel(),
                  iterations=pat)
    return result


def get_swap_target(target_idx, point, W, b=None):
    assert(point.shape[1] == 1)
    # Compute activation
    if b is not None:
        act = W.dot(point) + b
    else:
        act = W.dot(point)

    # If we just want to try to find a region where target_idx
    # is ranked as the max, we don't need to argsort act
    # We just need to detect what class is ranked above.
    argmax_idx = np.argmax(act) 
    return argmax_idx


def swap(ci, cj, cur_point, W, b=None):

    # Braid normal vector for pair(i, j)
    braid_vector = (W[ci, :] - W[cj, :]).reshape(-1, 1)
    if b is not None:
        braid_bias = b[ci] - b[cj]

    # Normalize normal vector to unit length
    # such that projection onto vector is dot product
    braid_norm = np.linalg.norm(braid_vector)
    braid_vector = braid_vector / braid_norm

    # Project the cur_point onto the normal vector
    # Note that braid_coeff will always be negative
    # unless we have found a solution
    braid_coeff = braid_vector.T.dot(cur_point)

    # Reflect the current point across the target braid hyperplane
    if b is None:
        reflection_on_hyperplane = cur_point - 2. * braid_coeff * braid_vector
    else:
        reflection_on_hyperplane = cur_point - 2. * (braid_coeff + braid_bias/braid_norm) * braid_vector

    return reflection_on_hyperplane