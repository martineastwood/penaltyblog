# penaltyblog/models/loss.pxd
import numpy as np

cimport numpy as np


# Expose the C-level function so other .pyx files can call it directly
cpdef double dixon_coles_loss_function(
    long[:] goals_home,
    long[:] goals_away,
    np.float64_t[:] weights,
    long[:] home_indices,
    long[:] away_indices,
    np.float64_t[:] attack,
    np.float64_t[:] defence,
    double hfa,
    double rho
) nogil
