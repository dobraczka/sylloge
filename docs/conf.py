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

from sphinx.ext.autodoc import between

from sylloge import __version__

sys.path.insert(0, os.path.abspath("."))


# -- use IGNORE for noqa in docstrings
def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    return app


# -- Project information -----------------------------------------------------

project = "sylloge"
copyright = "2022, Daniel Obraczka"  # noqa: A001
author = "Daniel Obraczka"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.automodapi",
]
automodsumm_inherited_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document. (see https://stackoverflow.com/a/56859983/6555620)
master_doc = "index"

autosummary_generate = True
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "insegel"
html_theme_options = {"navigation_depth": 2}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "logo.png"
