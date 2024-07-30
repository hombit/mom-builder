from concurrent.futures import ThreadPoolExecutor, as_completed

from mom_builder import MOMMerger, MOMBuilder


def gen_mom_from_fn(fn, *, max_norder, split_norder=None, merger, dtype=None,
                    tiles_consumer=lambda split_index, tiles: (split_index, tiles), n_threads=1):
    """Generator that builds a tree using the given function of healpix tile

    The generator goes over all subtrees and calls the given function with
    the max_norder and the indexes of the subtree, and yields the result of
    this subtree merge. Then it yields the result of the top tree merge.
    This generator is not thread-safe, and should not be sent across threads.

    See more details about the building process in the documentation of
    `MOMBuilder`.

    Parameters
    ----------
    fn : callable
        A function that takes the max_norder and the indexes of a subtree,
        and returns a MinMaxMeanState.
        The signature is:
        `fn(order:int, indexes:np.ndarray[uint64]) -> np.ndarray`
        Indexes are the healpix indexes of of the order `order`
        (always `max_norder`). The returned array is the value of the
        function for each of the indexes, it must has the same length as
        `indexes` and always be the same float dtype (so it should not return
        float32 for some indexes and int64 for others).
    merger : MOMMerger or float
        Merging algorithm to use. If a float is given, min-max-mean states
        are merged if the relative difference between the minimum and maximum
        values is below this threshold. It is the same as setting merger to
        MOMMerger("min-max-mean", "rtol", threshold=threshold, dtype=dtype),
        where dtype is derived from the return of `fn`.
    max_norder : int
        Maximum depth of the healpix tree.
    split_norder : int, optional
        The depth of the top tree. If not given, it is `max_norder // 2`,
        which should lead to a consistent memory usage. However, it can
        be suboptimal for performance of `fn`, consider set it to lower
        values to have a trade-off between memory and performance.
    tiles_consumer : Callable[int, tuple[int, np.ndarray, np.ndarray]], optional
        A function that takes the result of a subtree merge and the index of
        the subtree. The return value is yielded by the generator. It runs in
        the same thread as the subtree merger. The root tree merge would
        output -1 index. A possible use case is to write the tiles to a file.
        The default is the identity function.
    n_threads : int, optional
        Number of thread-jobs to use. Default is 1, which means no
        parallelism. If > 1, the builder will use a thread pool to
        build the subtrees in parallel.

    Yields
    ------
    list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
        List of (norder, indexes, values) tuples. The generator has two
        phases: first it yields the subtrees, then it yields the top tree.
        During the first phase, it yeilds the subtrees in the order of
        increasing index and, in an inner loop, in the order of increasing
        norder. That means that the same `norder` can be yielded multiple
        times, but always with increasing `indexes`.
    """
    if not isinstance(merger, MOMMerger):
        import numpy as np

        if dtype is None:
            dtype = fn(max_norder, np.arange(16, dtype=np.uint64)).dtype
        merger = MOMMerger("min-max-mean", "rtol", threshold=merger, dtype=dtype)

    if split_norder is None:
        split_norder = max_norder // 2

    # We need thread safety if we are going to use multiple threads
    thread_safe = n_threads > 1
    builder = MOMBuilder(merger, max_norder=max_norder, split_norder=split_norder, thread_safe=thread_safe)

    def worker(subtree_index):
        indexes = builder.subtree_maxnorder_indexes(subtree_index)
        values = fn(builder.max_norder, indexes)
        return tiles_consumer(subtree_index, builder.build_subtree(subtree_index, values))

    if n_threads == 1:
        gen_subtrees = _gen_subtrees_sequential
    else:
        gen_subtrees = gen_subtrees_parallel
    yield from gen_subtrees(
        fn,
        worker=worker,
        num_subtrees=builder.num_subtrees,
        n_threads=n_threads,
    )

    yield tiles_consumer(-1, builder.build_top_tree())


def _gen_subtrees_sequential(fn, *, worker, num_subtrees, n_threads):
    """Sequential version of `gen_mom_from_fn`"""
    del n_threads

    for subtree_index in range(num_subtrees):
        yield worker(subtree_index)


def gen_subtrees_parallel(fn, *, worker, num_subtrees, n_threads):
    """Parallel version of `gen_mom_from_fn`"""

    with ThreadPoolExecutor(n_threads) as executor:
        futures = [executor.submit(worker, subtree_index) for subtree_index in range(num_subtrees)]
        for future in as_completed(futures):
            yield future.result()
