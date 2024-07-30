import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mom_builder import mom_from_array, mom_from_batch_it
from mom_builder.mom_generator import gen_mom_from_fn


def tree_from_gen(g, max_norder):
    tree = [([], []) for _order in range(max_norder + 1)]
    for norder, indexes, values in g:
        tree[norder][0].append(indexes)
        tree[norder][1].append(values)
    for norder in range(max_norder + 1):
        tree[norder] = (np.concatenate(tree[norder][0]), np.concatenate(tree[norder][1]))
    return tree


def validate_tree_array_lengths(t, name):
    for norder, (indexes, values) in enumerate(t):
        assert len(indexes) == len(values), f"{name}, norder={norder}"
        assert len(indexes) <= 12 * 4 ** norder, f"{name}, norder={norder}"


def validate_tree_area(t, name):
    max_norder = len(t) - 1
    desired_area = 12 * 4 ** max_norder
    actual_area = sum(4 ** (max_norder - norder) * len(indexes) for norder, (indexes, _) in enumerate(t))
    assert actual_area == desired_area, f"{name}"


def validate_tree(t, name):
    validate_tree_array_lengths(t, name)
    validate_tree_area(t, name)


def compare_trees(t1, t2, name1, name2):
    assert len(t1) == len(t2)

    for norder in range(len(t1)):
        layer1, layer2 = t1[norder], t2[norder]
        assert_array_equal(layer1[0], layer2[0], err_msg=f"{name1} vs {name2} : indexes, norder={norder}")
        assert_array_equal(layer1[1], layer2[1], err_msg=f"{name1} vs {name2} : values, norder={norder}")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_arange(dtype):
    max_norder = 5
    ntiles = 12 * 4 ** max_norder
    threshold = 0.1
    split_norder = 3
    batch_size = 9

    data = np.arange(ntiles, dtype=dtype)

    tree_from_array = mom_from_array(data, max_norder, threshold=threshold)

    batch_iter = (data[i:i + batch_size] for i in range(0, ntiles, batch_size))
    tree_from_it = mom_from_batch_it(batch_iter, max_norder, threshold=threshold)

    gen = gen_mom_from_fn(
        fn=lambda _order, indexes: data[indexes],
        max_norder=max_norder,
        split_norder=split_norder,
        merger=threshold,
    )
    tree_from_fn = tree_from_gen(gen, max_norder)

    trees = [(tree_from_array, 'array'), (tree_from_it, 'it'), (tree_from_fn, 'fn')]
    for t, name in trees:
        validate_tree(t, name)

    for (t1, name1), (t2, name2) in zip(trees, trees[1:]):
        compare_trees(t1, t2, name1, name2)
