'''Utility constants and functions for parallelization'''
import joblib
import platform

N_JOBS = joblib.cpu_count() - int(.1 * joblib.cpu_count())
HOSTNAME = platform.uname()[1]


def is_local():
    for cpu in ['lintu', 'txori']:
        if cpu in HOSTNAME.lower():
            return True
    return False
