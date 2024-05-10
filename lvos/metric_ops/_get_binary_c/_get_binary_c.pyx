# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_b(np.ndarray[np.npy_bool, ndim=2] seg, np.ndarray[np.npy_bool, ndim=2] e, np.ndarray[np.npy_bool, ndim=2] s, np.ndarray[np.npy_bool, ndim=2] se):
    cdef int i, j
    cdef int m = seg.shape[0]
    cdef int n = seg.shape[1]
    cdef np.npy_bool seg_ij, e_s_se_ij
    cdef np.npy_bool[:, :] seg_view = seg
    cdef np.npy_bool[:, :] e_view = e
    cdef np.npy_bool[:, :] s_view = s
    cdef np.npy_bool[:, :] se_view = se
    cdef np.npy_bool[:, :] b_view = np.empty((m, n), dtype=np.bool_)

    for i in prange(m, nogil=True):
        for j in range(n):
            seg_ij = seg_view[i, j]
            e_s_se_ij = e_view[i, j] | s_view[i, j] | se_view[i, j]
            b_view[i, j] = seg_ij ^ e_s_se_ij

    return np.asarray(b_view)