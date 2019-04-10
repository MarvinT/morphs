from __future__ import absolute_import
from __future__ import division
import numpy as np
import scipy as sp


def f_poly(x, p, zero_center=True):
    if zero_center:
        # 1 -> -1, 128 -> 1
        x = (2 * x - 129) / 127
    return np.exp(np.sum([p_i * x ** i for i, p_i in enumerate(p)], axis=0))


def p0_poly(order, p_opt):
    bounds = (-np.inf, np.inf)
    if len(p_opt) + 1 == order:
        return np.append(p_opt, [1]), bounds
    else:
        return np.ones(order + 1), bounds


def integrate_intervals(p, sampled_points, f=f_poly):
    return np.array(
        [
            sp.integrate.quad(f, a, b, args=(p,))[0]
            for a, b in zip(sampled_points[:-1], sampled_points[1:])
        ]
    )


def create_curve_fit_f(sampled_points, f=f_poly):
    def curve_fit_f(contains_interval, *p):
        interval_vals = integrate_intervals(p, sampled_points, f=f)
        return contains_interval.dot(interval_vals)

    return curve_fit_f


def fit_derivative(
    group, p0, y_label="red_neural_cosine_dist", f=f_poly, bounds=(-np.inf, np.inf)
):
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
        return sp.optimize.curve_fit(
            curve_fit_f, contains_interval, y, p0=p0, bounds=bounds
        )
    except RuntimeError:
        return np.nan * p0, None
