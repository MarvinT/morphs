"""Utility constants and functions for parallelization"""
from __future__ import absolute_import
import joblib
import platform

HOSTNAME = platform.uname()[1]
N_JOBS = joblib.cpu_count()


def is_local():
    for cpu in ["lintu", "txori"]:
        if cpu in HOSTNAME.lower():
            return True
    return False


if is_local():
    N_JOBS -= int(0.1 * joblib.cpu_count())
