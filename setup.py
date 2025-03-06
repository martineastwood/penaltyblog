import os
import platform
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


class BuildGoExtension(build_ext):
    """Custom build command to compile Go shared library before packaging."""

    def run(self):
        root = os.path.dirname(os.path.abspath(__file__))
        gosrc = os.path.join(root, "penaltyblog", "gosrc")
        output_dir = os.path.join(root, "penaltyblog", "golib")

        # Ensure golib directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Detect OS and build the correct shared library format
        if platform.system() == "Windows":
            output_file = os.path.join(output_dir, "penaltyblog.dll")
            build_cmd = ["go", "build", "-o", output_file, "-buildmode=c-shared", "."]
        elif platform.system() == "Darwin":
            output_file = os.path.join(output_dir, "penaltyblog.dylib")
            build_cmd = ["go", "build", "-o", output_file, "-buildmode=c-shared", "."]
        else:
            output_file = os.path.join(output_dir, "penaltyblog.so")
            build_cmd = ["go", "build", "-o", output_file, "-buildmode=c-shared", "."]

        print(f"🚀 Building Go shared library: {output_file}")
        subprocess.run(build_cmd, cwd=gosrc, check=True)

        super().run()


class CustomInstall(install):
    """Custom install command that forces build_ext to run first."""

    def run(self):
        self.run_command("build_ext")
        super().run()


setup(
    name="penaltyblog",
    version="1.1.0",
    packages=find_packages(include=["penaltyblog*"]),
    include_package_data=True,
    # Adding an empty ext_modules forces build_ext to run
    ext_modules=[],
    cmdclass={
        "build_ext": BuildGoExtension,
        "install": CustomInstall,
    },
)
