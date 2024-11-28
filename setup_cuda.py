from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os
import platform
#import torch
import torch.utils.cpp_extension

# Get PyTorch's include paths including CUDA
torch_include_dirs = torch.utils.cpp_extension.include_paths()

# Get CUDA include paths
cuda_include_dirs = []
if torch.cuda.is_available():
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is not None:
        cuda_include_dirs = [os.path.join(cuda_home, "include")]

# Combine all include paths
include_dirs = (
    #torch_include_dirs +
    cuda_include_dirs + 
    [os.path.dirname(torch.__file__),
     os.path.join(os.path.dirname(torch.__file__), "include")]
)

# Get library directories
torch_library_dirs = [os.path.join(os.path.dirname(torch.__file__), "lib")]
if cuda_home:
    torch_library_dirs.append(os.path.join(cuda_home, "lib64"))

# Paths
ompeval_dir = os.path.abspath("./OMPEval")
ompeval_include_dir = ompeval_dir 
ompeval_lib_dir = os.path.join(ompeval_dir, "lib")
ompeval_lib_path = os.path.join(ompeval_lib_dir, "libompeval.a")

project_root = os.path.abspath(".")
src_dir = os.path.join(project_root, "src")

# Print paths for debugging
print("Include dirs:", include_dirs)
print("Library dirs:", torch_library_dirs)

ext_modules = [
    Pybind11Extension(
        "poker_inference",
        ["lib/poker_inference_binding.cpp"],
        include_dirs=[
            *include_dirs,
            ompeval_include_dir,
            src_dir,
        ],
        library_dirs=[
            *torch_library_dirs,
            ompeval_lib_dir,
        ],
        libraries=['torch_cuda', 'c10', 'c10_cuda'],
        extra_objects=[ompeval_lib_path],
        extra_compile_args=[
            "-std=c++17",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=poker_inference",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
        runtime_library_dirs=[*torch_library_dirs],
    ),
    Pybind11Extension(
        "ompeval",
        ["lib/eval_binding.cpp"],
        include_dirs=[ompeval_include_dir, src_dir],
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
    description="Python bindings for OMPEval and Poker Inference",
    ext_modules=ext_modules,
)