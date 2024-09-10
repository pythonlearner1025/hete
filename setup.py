from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import os

# Assume OMPEval is in a subdirectory of your project called 'OMPEval'
# Adjust this path if it's located elsewhere
ompeval_dir = os.path.abspath("./OMPEval")
ompeval_include_dir = ompeval_dir 
ompeval_lib_dir = os.path.join(ompeval_dir, "lib")
ompeval_lib_path = os.path.join(ompeval_lib_dir, "ompeval.a")  # or libompeval.dylib if it's a shared library

ext_modules = [
    Pybind11Extension(
        "ompeval",
        ["lib/eval_binding.cpp"],
        include_dirs=[ompeval_include_dir],
        library_dirs=[ompeval_lib_dir],
        extra_objects=[ompeval_lib_path],
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + ompeval_lib_dir],
    ),
]

setup(
    name="ompeval",
    version="0.0.1",
    author="Your Name",
    description="Python bindings for OMPEval",
    ext_modules=ext_modules,
)