# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False

from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp

cdef extern from "fofW.h":
    cdef cppclass fofW:
        fofW(vector[float] X_locations, vector[float] Y_locations, vector[float] SNR, vector[float] Widths)
        void do_clustering()
        vector[vector[size_t]] final_clusters()

cdef class PyfofW:
    cdef fofW* _c_clustering

    def __init__(self, object x_locations, object y_locations, object snr, object widths):
        cdef cnp.ndarray[float, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x_locations, dtype=np.float32)
        cdef cnp.ndarray[float, ndim=1, mode="c"] y_arr = np.ascontiguousarray(y_locations, dtype=np.float32)
        cdef cnp.ndarray[float, ndim=1, mode="c"] s_arr = np.ascontiguousarray(snr, dtype=np.float32)
        cdef cnp.ndarray[float, ndim=1, mode="c"] w_arr = np.ascontiguousarray(widths, dtype=np.float32)

        cdef size_t n = x_arr.shape[0]

        cdef vector[float] c_x = vector[float](&x_arr[0], &x_arr[0] + n)
        cdef vector[float] c_y = vector[float](&y_arr[0], &y_arr[0] + n)
        cdef vector[float] c_snr = vector[float](&s_arr[0], &s_arr[0] + n)
        cdef vector[float] c_widths = vector[float](&w_arr[0], &w_arr[0] + n)

        self._c_clustering = new fofW(c_x, c_y, c_snr, c_widths)

    def do_clustering(self):
        self._c_clustering.do_clustering()

    def final_clusters(self):
        # Explicitly convert C++ vector<vector<size_t>> into a Python list of lists
        cdef vector[vector[size_t]] c_res = self._c_clustering.final_clusters()
        py_clusters = []
        
        for i in range(c_res.size()):
            inner_vec = c_res[i]
            py_inner = [inner_vec[j] for j in range(inner_vec.size())]
            py_clusters.append(py_inner)
            
        return py_clusters

    def __dealloc__(self):
        if self._c_clustering != NULL:
            del self._c_clustering
