'''
@author: cbwces
@date: 20210419
@contact: sknyqbcbw@gmail.com
'''
cimport cython
from cython.parallel import prange
import numpy
cimport numpy
from libc.math cimport ceil, floor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int MAX(int a, int b):
    if a > b:
        b = a
    return b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int MIN(int a, int b):
    if a < b:
        b = a
    return b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef (double, double) minmax(numpy.ndarray[double, ndim=1, mode='c'] arr):
    cdef double min_ = 999999.
    cdef double max_ = -999999.
    cdef Py_ssize_t i
    for i in range(arr.shape[0]):
        if arr[i] < min_:
            min_ = arr[i]
        if arr[i] > max_:
            max_ = arr[i]
    return min_, max_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def render_cy(numpy.ndarray[double, ndim=2, mode='c'] vertices, numpy.ndarray[double, ndim=2, mode='c'] new_colors, numpy.ndarray[long, ndim=2, mode='c'] triangles, int h, int w):
    cdef Py_ssize_t vertices_shape0 = vertices.shape[1]
    cdef numpy.ndarray[double, ndim=2, mode='c'] vis_colors = numpy.ones((1, vertices_shape0))
    cdef numpy.ndarray[double, ndim=3, mode='c'] face_mask = render_texture(vertices, vis_colors, triangles, h, w, 1)
    cdef numpy.ndarray[double, ndim=3, mode='c'] new_image = render_texture(vertices, new_colors, triangles, h, w, 3)
    return face_mask, new_image

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef numpy.ndarray[double, ndim=3, mode='c'] render_texture(numpy.ndarray[double, ndim=2, mode='c'] vertices,  numpy.ndarray[double, ndim=2, mode='c'] colors, numpy.ndarray[long, ndim=2, mode='c'] triangles, int h, int w, int c = 3):

    cdef numpy.ndarray[double, ndim=3, mode='c'] image = numpy.empty((h, w, c), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=2, mode='c'] depth_buffer = numpy.zeros([h, w], dtype=numpy.double) - 999999.

    cdef Py_ssize_t triangles_size_0 = triangles.shape[0]
    cdef Py_ssize_t triangles_size_1 = triangles.shape[1]
    cdef Py_ssize_t triangles_size_0_ptr
    cdef Py_ssize_t triangles_size_1_ptr

    cdef Py_ssize_t colors_size = colors.shape[0]
    cdef Py_ssize_t colors_size_ptr

    cdef numpy.ndarray[double, ndim=1, mode='c'] tri_depth = numpy.empty((triangles_size_1), dtype=numpy.double)
    cdef numpy.ndarray[double, ndim=2, mode='c'] tri_tex = numpy.empty((colors_size, triangles_size_1), dtype=numpy.double)

    for triangles_size_1_ptr in prange(triangles_size_1, nogil=True):
        tri_depth[triangles_size_1_ptr] = (vertices[2, triangles[0, triangles_size_1_ptr]] + vertices[2, triangles[1, triangles_size_1_ptr]] + vertices[2, triangles[2, triangles_size_1_ptr]]) / 3.
        for colors_size_ptr in range(colors_size):
            tri_tex[colors_size_ptr, triangles_size_1_ptr] = (colors[colors_size_ptr, triangles[0, triangles_size_1_ptr]] + colors[colors_size_ptr, triangles[1, triangles_size_1_ptr]] + colors[colors_size_ptr, triangles[2, triangles_size_1_ptr]]) / 3.

    cdef int umin
    cdef int vmin
    cdef int umax
    cdef int vmax
    cdef Py_ssize_t u
    cdef Py_ssize_t v
    cdef double relate_min
    cdef double relate_max
    cdef numpy.ndarray[long, ndim=1, mode='c'] tri = numpy.empty((triangles_size_0,), dtype=numpy.long)
    cdef Py_ssize_t c_channel_ptr
    cdef numpy.ndarray[double, ndim=2, mode='c'] vertices_idx_by_tri = numpy.empty((2, triangles_size_0), dtype=numpy.double)
    cdef bint ifisPointInTri

    for triangles_size_1_ptr in range(triangles_size_1):
        for triangles_size_0_ptr in range(triangles_size_0):
            tri[triangles_size_0_ptr] = triangles[triangles_size_0_ptr, triangles_size_1_ptr]
            vertices_idx_by_tri[0, triangles_size_0_ptr] = vertices[0, tri[triangles_size_0_ptr]]
            vertices_idx_by_tri[1, triangles_size_0_ptr] = vertices[1, tri[triangles_size_0_ptr]]

        relate_min, relate_max = minmax(vertices_idx_by_tri[0])

        umin = MAX(<int>(ceil(relate_min)), 0)
        umax = MIN(<int>(floor(relate_max)), w-1)

        relate_min, relate_max = minmax(vertices_idx_by_tri[1])
        vmin = MAX(<int>(ceil(relate_min)), 0)
        vmax = MIN(<int>(floor(relate_max)), h-1)

        if umax<umin or vmax<vmin:
            continue
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[triangles_size_1_ptr] > depth_buffer[v, u]:
                    ifisPointInTri = isPointInTri(<double>u, <double>v, vertices_idx_by_tri)
                    if ifisPointInTri:
                        depth_buffer[v, u] = tri_depth[triangles_size_1_ptr]
                        for c_channel_ptr in range(c):
                            image[v, u, c_channel_ptr] = tri_tex[c_channel_ptr, triangles_size_1_ptr]
    return image

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef bint isPointInTri(double point0, double point1, numpy.ndarray[double, ndim=2, mode='c'] tp0):

    cdef double dot00 = 0
    cdef double dot01 = 0
    cdef double dot02 = 0
    cdef double dot11 = 0
    cdef double dot12 = 0

    dot00 += (tp0[0, 2]-tp0[0, 0])*(tp0[0, 2]-tp0[0, 0])
    dot00 += (tp0[1, 2]-tp0[1, 0])*(tp0[1, 2]-tp0[1, 0])
    dot01 += (tp0[0, 2]-tp0[0, 0])*(tp0[0, 1]-tp0[0, 0])
    dot01 += (tp0[1, 2]-tp0[1, 0])*(tp0[1, 1]-tp0[1, 0])
    dot02 += (tp0[0, 2]-tp0[0, 0])*(point0-tp0[0, 0])
    dot02 += (tp0[1, 2]-tp0[1, 0])*(point1-tp0[1, 0])
    dot11 += (tp0[0, 1]-tp0[0, 0])*(tp0[0, 1]-tp0[0, 0])
    dot11 += (tp0[1, 1]-tp0[1, 0])*(tp0[1, 1]-tp0[1, 0])
    dot12 += (tp0[0, 1]-tp0[0, 0])*(point0-tp0[0, 0])
    dot12 += (tp0[1, 1]-tp0[1, 0])*(point1-tp0[1, 0])

    cdef double inverDeno

    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0.0
    else:
        inverDeno = 1.0/(dot00*dot11 - dot01*dot01)

    cdef double u = (dot11*dot02 - dot01*dot12)*inverDeno
    cdef double v = (dot00*dot12 - dot01*dot02)*inverDeno

    return (u >= 0) & (v >= 0) & (u + v < 1)
