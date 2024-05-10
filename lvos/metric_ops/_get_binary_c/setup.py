from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("_get_binary_c.pyx"),
    include_dirs=[numpy.get_include()]
)