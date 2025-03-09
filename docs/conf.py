# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))
# sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = "penaltyblog"
copyright = "2021, Martin Eastwood"
author = "Martin Eastwood"

# source_suffix = ".rst"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

master_doc = "index"

html_logo = "_static/logo.png"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


html_sidebars = {
    "**": [
        "globaltoc.html",
    ],
}

html_theme_options = {
    "pygment_light_style": "tango",
    "pygment_dark_style": "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/martineastwood/penaltyblog",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

# autosummary_generate = ["api_reference.rst"]

html_static_path = ["_static"]

html_permalinks_icon = "<span>#</span>"
html_theme = "pydata_sphinx_theme"

pygments_style = "sphinx"
