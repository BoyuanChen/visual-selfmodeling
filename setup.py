try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'utils.libkdtree.pykdtree.kdtree',
    sources=[
        'utils/libkdtree/pykdtree/kdtree.c',
        'utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)


# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'utils.libmesh.triangle_hash',
    sources=[
        'utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)


# Gather all extension modules
ext_modules = [
    pykdtree,
    triangle_hash_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_ext': build_ext
    }
)