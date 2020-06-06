# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Imports -----------------------------------------------------------------

import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'AIToolbox'
copyright = f'{datetime.datetime.now().year}, Marko Vidoni'
author = 'Marko Vidoni'

# The full version, including alpha/beta/rc tags
version = '1.1.0'
release = '1.1.0'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

add_module_names = True
add_function_parentheses = True

autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Handled already by the `autoclass_content` above in a nicer way
# autodoc_default_options = {
#     'special-members': '__init__'
# }

autodoc_mock_imports = ['tensorflow', 'keras']

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.apidoc'  # Using https://github.com/sphinx-contrib/apidoc
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# For documentation of `sphinxcontrib.apidoc` params have a look at:
#   https://github.com/sphinx-contrib/apidoc
apidoc_module_dir = '../../aitoolbox'
apidoc_output_dir = 'api'
apidoc_excluded_paths = ['kerastrain', 'tftrain']
apidoc_separate_modules = True
apidoc_module_first = True

apidoc_extra_args = ['-f', '-t', f'{os.path.dirname(__file__)}/_templates/apidoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', '_templates/apidoc']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    # these are for sphinx_rtd_theme:
    'collapse_navigation': False,
    'navigation_depth': -1,
    'titles_only': True

    # these are for alabaster:
    # 'show_relbars': True,
    # 'fixed_sidebar': True,
    # 'sidebar_collapse': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
