# setup.py
import glob
import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


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

# Process .pyx files in penaltyblog/bayes
bayes_pyx = find_pyx_files("penaltyblog", "bayes")
for pyx_path in bayes_pyx:
    module_name = os.path.splitext(pyx_path.replace(os.sep, "."))[0]
    extensions.append(
        Extension(
            module_name,
            [pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    )


class build_py(_build_py):
    """Remove stale xT files from reused build directories."""

    def run(self):
        super().run()

        stale_paths = [
            Path(self.build_lib) / "penaltyblog" / "xt" / "data.py",
            Path(self.build_lib) / "penaltyblog" / "xt" / "data" / "xt_default_v1.npz",
        ]
        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()


setup(
    name="penaltyblog",
    version="1.12.0",
    description="Library from http://pena.lt/y/blog for scraping and modelling football (soccer) data",
    packages=find_packages(
        include=["penaltyblog", "penaltyblog.*"],
        exclude=["penaltyblog.test", "penaltyblog.docs"],
    ),
    cmdclass={"build_py": build_py},
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
        "wrapper-tls-requests",
    ],
)
