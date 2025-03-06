import ctypes

import numpy as np

from . import go_lib

go_lib.ComputeAverageRPS.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # probs (flattened 2D array)
    ctypes.POINTER(ctypes.c_int),  # outcomes (as C.int)
    ctypes.c_int,  # nSets
    ctypes.c_int,  # nOutcomes
]
go_lib.ComputeAverageRPS.restype = ctypes.c_double

go_lib.ComputeRPSArray.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # probs (flattened)
    ctypes.POINTER(ctypes.c_int),  # outcomes (1D)
    ctypes.c_int,  # nSets
    ctypes.c_int,  # nOutcomes
    ctypes.POINTER(ctypes.c_double),  # out (output array)
]
go_lib.ComputeRPSArray.restype = None
