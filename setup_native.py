"""Build the optional native extension in place."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="mechanica-native",
    ext_modules=[
        CppExtension(
            "mechanica._mechanica_native",
            ["src/mechanica/native/spring.cpp", "src/mechanica/native/robotics.cpp"],
            extra_compile_args={"cxx": ["/O2"] if os.name == "nt" else ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "src"},
)
