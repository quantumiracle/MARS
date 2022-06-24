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
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../"))  # Important
sys.path.insert(0, os.path.abspath("../.."))  # Important
sys.path.insert(0, os.path.abspath(os.path.join("../..", "mars")))  # Important
# print(sys.path)

# -- Project information -----------------------------------------------------

project = 'MARS'
copyright = '2021, Zihan Ding'
author = 'Zihan Ding'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',  # parse md as well
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    # 'sphinx.ext.imgmath',
    # 'sphinx.ext.mathjax',
    # 'sphinx_math_dollar',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # 'sphinxcontrib.bibtex',
    # 'recommonmark'
]

myst_enable_extensions = ["dollarmath", "amsmath"]  # https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#math-shortcuts


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    'cv2',
    'hyperdash',
    'gridfs',
    'horovod',
    'hyperdash',
    'imageio',
    'lxml',
    'matplotlib',
    'nltk',
    'cvxpy',
    'PIL',
    'progressbar',
    'pymongo',
    'scipy',
    'skimage',
    'sklearn',
    # 'tensorflow',
    'tqdm',
    'h5py',
    # 'tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops',  # TL C++ Packages
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = 'img/mars_label.jpg'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
