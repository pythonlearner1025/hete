from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import os

# Paths
ompeval_dir = os.path.abspath("./OMPEval")
ompeval_include_dir = ompeval_dir 
ompeval_lib_dir = os.path.join(ompeval_dir, "lib")
ompeval_lib_path = os.path.join(ompeval_lib_dir, "libompeval.a")

project_root = os.path.abspath(".")
src_dir = os.path.join(project_root, "src")

# Path to libtorch - adjust this to your libtorch location
libtorch_path = "/Users/minjunes/libtorch"
libtorch_include_dir = os.path.join(libtorch_path, "include")
libtorch_lib_dir = os.path.join(libtorch_path, "lib")

ext_modules = [
    Pybind11Extension(
        "ompeval",
        ["lib/eval_binding.cpp"],
        include_dirs=[ompeval_include_dir, src_dir],
        library_dirs=[ompeval_lib_dir],
        extra_objects=[ompeval_lib_path],
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-Wl,-rpath," + ompeval_lib_dir],
    ),
    Pybind11Extension(
        "poker_inference",
        ["lib/poker_inference_binding.cpp"],
        include_dirs=[
            ompeval_include_dir,
            src_dir,
            libtorch_include_dir,
            os.path.join(libtorch_include_dir, "torch", "csrc", "api", "include"),
        ],
        library_dirs=[
            ompeval_lib_dir,
            libtorch_lib_dir,
        ],
        extra_objects=[ompeval_lib_path],
        extra_compile_args=[
            "-std=c++17",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=poker_inference",
        ],
        extra_link_args=[
            "-Wl,-rpath," + ompeval_lib_dir,
            "-Wl,-rpath," + libtorch_lib_dir,
            "-ltorch",
            "-lc10",
        ],
    ),
]

setup(
    name="ompeval",
    version="0.0.1",
    author="Your Name",
    description="Python bindings for OMPEval and Poker Inference",
    ext_modules=ext_modules,
)
