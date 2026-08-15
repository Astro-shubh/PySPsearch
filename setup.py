from Cython.Build import cythonize
from setuptools import Extension, setup
import os
import numpy as np

conda_prefix = os.environ.get("CONDA_PREFIX", "")
conda_include = os.path.join(conda_prefix, "include") if conda_prefix else "/usr/include"

extensions = [
    Extension(
        "modules.fofw_wrap",  # <-- Updated module name here
        sources=[
            "modules/fofw_wrap.pyx",
            "modules/fofW.cpp",
        ],
        include_dirs=[
            "modules",
            np.get_include(),
            conda_include,
        ],
        extra_compile_args=["-O3", "-march=native", "-std=c++17"],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3})
)
