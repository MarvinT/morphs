from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit
import morphs
import pickle
from joblib import Parallel, delayed
import click
import datetime
import resource


def f_poly(x, p, zero_center=True):
    if zero_center:
        # 1 -> -1, 128 -> 1
        x = (2 * x - 129) / 127
    return np.exp(np.sum([p_i * x ** i for i, p_i in enumerate(p)], axis=0))


def p0_poly(order, *args):
    bounds = (-np.inf, np.inf)
    if len(args) > 0:
        p_opt = args[0]
        if np.all(np.isfinite(p_opt)) and len(p_opt) + 1 == order:
            return np.append(p_opt, [1]), bounds
    return np.ones(order + 1), bounds


def f_map(popt, x=np.linspace(1, 128), normalize=True, f=f_poly):
    y = f(x, popt)
    if normalize:
        y /= quad(f, 1, 128, args=(popt,))[0]
    return y


def integrate_pairs(p, pairs, f=f_poly):
    return np.array([quad(f, a, b, args=(p,))[0] for a, b in pairs])


def integrate_intervals(p, sampled_points, f=f_poly):
    return integrate_pairs(p, zip(sampled_points[:-1], sampled_points[1:]), f=f)


def create_curve_fit_f(sampled_points, f=f_poly):
    def curve_fit_f(contains_interval, *p):
        interval_vals = integrate_intervals(p, sampled_points, f=f)
        return contains_interval.dot(interval_vals)

    return curve_fit_f


def fit_derivative(
    group, p0, y_label="red_neural_cosine_dist", f=f_poly, bounds=(-np.inf, np.inf)
):
    group = group.dropna(subset=[y_label])
    y = group[y_label].values
    sampled_points = group["lesser_morph_pos"].unique()
    assert np.all(np.diff(sampled_points) >= 0), "sampled points not sorted"
    contains_interval = np.zeros((len(group), len(sampled_points) - 1), dtype=bool)
    boundaries = (sampled_points[:-1] + sampled_points[1:]) / 2.0
    for i, boundary in enumerate(boundaries):
        contains_interval[:, i] = (group["greater_morph_pos"] > boundary) & (
            boundary > group["lesser_morph_pos"]
        ).values

    curve_fit_f = create_curve_fit_f(sampled_points, f=f)
    try:
        return curve_fit(curve_fit_f, contains_interval, y, p0=p0, bounds=bounds)
    except RuntimeError:
        return np.nan * p0, None


def fit_all_derivatives(group, idxs=range(7), p0_func=p0_poly, **kwargs):
    deriv_dict = {}
    popt = np.array([])
    for i in idxs:
        p0, bounds = p0_func(i, popt)
        popt, pcov = fit_derivative(group, p0=p0, bounds=bounds, **kwargs)
        deriv_dict[i] = popt
    return deriv_dict


def _par_fad(group, block_path, morph_dim, max_order, **kwargs):
    """a wrapper for parallelization of fit_all_derivatives"""
    return (
        fit_all_derivatives(group, idxs=range(max_order), **kwargs),
        block_path,
        morph_dim,
    )


def fit_next_derivatives(group, i, prev_popt, p0_func=p0_poly, **kwargs):
    p0, bounds = p0_func(i, prev_popt)
    popt, pcov = fit_derivative(group, p0=p0, bounds=bounds, **kwargs)
    return i, popt


def _par_fnd(group, i, prev_popt, block_path, morph_dim, **kwargs):
    """a wrapper for parallelization of fit_next_derivatives"""
    return (
        fit_next_derivatives(group, i, prev_popt, **kwargs),
        block_path,
        morph_dim,
    )


def gen_derivative_dict(parallel=True, n_jobs=morphs.parallel.N_JOBS, max_order=7):
    pair_df = morphs.load.pop_pair_df()
    morphs.data.parse.morph_dim(pair_df)

    if parallel and n_jobs > 1:
        all_dds = Parallel(n_jobs=n_jobs)(
            delayed(_par_fad)(group, block_path, morph_dim, max_order)
            for (block_path, morph_dim), group in pair_df.groupby(
                ["block_path", "morph_dim"]
            )
        )
    else:
        all_dds = [
            _par_fad(group, block_path, morph_dim, max_order)
            for (block_path, morph_dim), group in pair_df.groupby(
                ["block_path", "morph_dim"]
            )
        ]

    deriv_dict = {block: {} for block in pair_df["block_path"].unique()}
    for dd, block_path, morph_dim in all_dds:
        deriv_dict[block_path][morph_dim] = dd

    morphs.paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(morphs.paths.DERIVATIVE_PKL.as_posix(), "wb") as f:
        pickle.dump(deriv_dict, f)


def increment_derivative_dict(
    pair_df=None, parallel=True, n_jobs=morphs.parallel.N_JOBS
):
    deriv_dict = load_derivative_dict()

    if pair_df is None:
        pair_df = morphs.load.pop_pair_df()
        morphs.data.parse.morph_dim(pair_df)

    prev_i = find_max_order(deriv_dict)

    if parallel and n_jobs > 1:
        all_ddus = Parallel(n_jobs=n_jobs)(
            delayed(_par_fnd)(
                group,
                prev_i + 1,
                deriv_dict[block_path][morph_dim][prev_i],
                block_path,
                morph_dim,
            )
            for (block_path, morph_dim), group in pair_df.groupby(
                ["block_path", "morph_dim"]
            )
        )
    else:
        all_ddus = [
            _par_fnd(
                group,
                prev_i + 1,
                deriv_dict[block_path][morph_dim][prev_i],
                block_path,
                morph_dim,
            )
            for (block_path, morph_dim), group in pair_df.groupby(
                ["block_path", "morph_dim"]
            )
        ]

    for (i, popt), block_path, morph_dim in all_ddus:
        deriv_dict[block_path][morph_dim][i] = popt

    with open(morphs.paths.DERIVATIVE_PKL.as_posix(), "wb") as f:
        pickle.dump(deriv_dict, f)


def find_max_order(deriv_dict):
    return np.max(
        list(
            {
                i
                for block_path in deriv_dict
                for morph_dim in deriv_dict[block_path]
                for i in deriv_dict[block_path][morph_dim]
            }
        )
    )


@morphs.utils.load._load(morphs.paths.DERIVATIVE_PKL, gen_derivative_dict)
def load_derivative_dict():
    return morphs.utils.load._pickle(morphs.paths.DERIVATIVE_PKL)


def load_derivative_df(melt=False):
    dd = morphs.load.derivative_dict()
    ddf = pd.DataFrame.from_dict(
        {(i, j): dd[i][j] for i in dd for j in dd[i]}, orient="index"
    )
    ddf.index.rename(["block_path", "morph_dim"], inplace=True)
    ddf.reset_index(inplace=True)
    if melt:
        ddf = pd.melt(
            ddf,
            id_vars=["block_path", "morph_dim"],
            var_name="order",
            value_name="popt",
        )
    morphs.data.parse.morph_dim(ddf)
    return ddf


@click.command()
@click.option(
    "--parallel",
    "-p",
    is_flag=True,
    help="whether to parallelize each morph_dim to its own process",
)
@click.option(
    "--num_jobs", default=morphs.parallel.N_JOBS, help="number of parallel cores to use"
)
@click.option(
    "--max_order", default=7, help="(not including) max polynomial order to fit"
)
def _main(parallel, num_jobs, max_order):
    tstart = datetime.datetime.now()

    if not morphs.paths.DERIVATIVE_PKL.exists():
        if max_order > 7:
            prev_max_order = 6
            needed_increments = max_order - prev_max_order - 1
            max_order = 7
        else:
            needed_increments = 0
        gen_derivative_dict(parallel=parallel, n_jobs=num_jobs, max_order=max_order)
    else:
        deriv_dict = load_derivative_dict()
        prev_max_order = find_max_order(deriv_dict)
        needed_increments = max_order - prev_max_order - 1

    if needed_increments > 0:
        pair_df = morphs.load.pop_pair_df()
        morphs.data.parse.morph_dim(pair_df)
        print("prev max order: ", prev_max_order)
        for i in range(needed_increments):
            increment_derivative_dict(
                pair_df=pair_df, parallel=parallel, n_jobs=num_jobs
            )
            print("max order incremented!")

    print(
        "peak memory usage: %f GB"
        % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    )
    print("time: %s" % (datetime.datetime.now() - tstart))


if __name__ == "__main__":
    _main()
