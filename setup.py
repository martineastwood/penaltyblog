# setup.py
import glob
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def find_pyx_files(package, subdir):
    """Return a list of .pyx files in a given subdirectory of a package."""
    pattern = os.path.join(package, subdir, "*.pyx")
    return glob.glob(pattern)


extensions = []

# Process .pyx files in penaltyblog/models
models_pyx = find_pyx_files("penaltyblog", "models")
for pyx_path in models_pyx:
    # Convert file path to module name by replacing OS separators with dots and removing .pyx extension
    module_name = os.path.splitext(pyx_path.replace(os.sep, "."))[0]
    extensions.append(
        Extension(
            module_name,
            [pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    )

# Process .pyx files in penaltyblog/metrics
rps_pyx = find_pyx_files("penaltyblog", "metrics")
for pyx_path in rps_pyx:
    module_name = os.path.splitext(pyx_path.replace(os.sep, "."))[0]
    extensions.append(
        Extension(
            module_name,
            [pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    )

setup(
    name="penaltyblog",
    version="1.5.0",
    description="Library from http://pena.lt/y/blog for scraping and modelling football (soccer) data",
    packages=find_packages(
        include=["penaltyblog", "penaltyblog.*"],
        exclude=["penaltyblog.test", "penaltyblog.docs"],
    ),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    install_requires=[
        "beautifulsoup4",
        "cssselect",
        "cython",
        "html5lib",
        "ipywidgets",
        "kaleido",
        "lxml",
        "matplotlib",
        "networkx",
        "numpy",
        "orjson",
        "pandas",
        "plotly",
        "PuLP",
        "requests",
        "scipy",
        "statsbombpy",
        "tabulate",
        "tqdm",
    ],
)
