"""Build the es4mgh C extension. Metadata lives in pyproject.toml."""

import sys

from setuptools import Extension, setup

extra_compile = ["-O3", "-std=c11", "-Wall"]
extra_link = []
if sys.platform == "darwin":
    extra_compile += ["-fblocks"]
    extra_link += ["-framework", "Accelerate"]

setup(
    ext_modules=[
        Extension(
            "es4mgh._native",
            sources=["src/es4mgh/_native.c", "es4mgh.c"],
            include_dirs=["."],
            extra_compile_args=extra_compile,
            extra_link_args=extra_link,
            libraries=["m"],
        )
    ],
)
