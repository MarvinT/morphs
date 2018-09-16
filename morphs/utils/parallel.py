'''Utility constants and functions for parallelization'''
import joblib

N_JOBS = joblib.cpu_count() - int(.1 * joblib.cpu_count())
