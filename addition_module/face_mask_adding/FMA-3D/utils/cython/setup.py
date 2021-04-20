'''
@author: cbwces
@date: 20210419
@contact: sknyqbcbw@gmail.com
'''
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "render",
        ["render.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
