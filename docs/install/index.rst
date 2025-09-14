Installation
============

``penaltyblog`` can be easily installed from PyPI or built from source. The package is optimized using **Cython**, which requires a C compiler if building from source. However, pre-built wheels are available on PyPI for most modern Python versions and operating systems (Windows, macOS, Linux).

Installing from PyPI (recommended)
----------------------------------

To install the latest stable release:

.. code-block:: bash

    pip install penaltyblog

If you want to read / write data from cloud storage (e.g., AWS S3, Google Cloud Storage) using ``Matchflow``,
you may need to install additional dependencies:

.. code-block:: bash

    pip install penaltyblog[s3]      # For AWS S3 support
    pip install penaltyblog[gcs]     # For Google Cloud Storage support
    pip install penaltyblog[azure]   # For Azure Blob Storage support
    pip install penaltyblog[cloud]   # For all cloud storage support

Installing from Source
----------------------

To install from source, clone the repository and install dependencies:

.. code-block:: bash

    git clone https://github.com/martineastwood/penaltyblog.git
    cd penaltyblog
    pip install .

Ensure you have a suitable C compiler available (e.g., GCC, Clang, or MSVC) when building from source.
