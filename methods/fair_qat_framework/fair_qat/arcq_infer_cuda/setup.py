import os

os.environ.setdefault("CUDA_HOME", "/gpool/opt/cuda/11.8")
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="arcq_infer_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="arcq_infer_cuda_ext",
            sources=[
                "arcq_infer_bind.cpp",
                "arcq_pack_kernel.cu",
                "arcq_activation_kernel.cu",
                "arcq_linear_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
