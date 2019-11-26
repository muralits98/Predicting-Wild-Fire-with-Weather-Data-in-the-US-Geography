from setuptools import setup, Extension
import numpy

multiordinal = Extension("multiordinal", sources=["src/multiordinal.c"],
                         include_dirs=[numpy.get_include()])
setup(ext_modules=[multiordinal])
