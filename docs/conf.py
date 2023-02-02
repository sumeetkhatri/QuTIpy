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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------


import qutipy
import importlib
from datetime import date

project = 'QuTIpy'
copyright = f'{date.today().year}, Sumeet Khatri'
author = 'Sumeet Khatri'

# The full version, including alpha/beta/rc tags
release = '0.1'
version = qutipy.__version__
language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'numpydoc',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    # 'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'hoverxref.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',


    'sphinx.ext.coverage',
    'sphinx.ext.graphviz',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx_panels',
    'sphinxemoji.sphinxemoji',
]
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    # 'neps': ('https://numpy.org/neps', None),
    # 'python': ('https://docs.python.org/3', None),
    # 'scipy': ('https://docs.scipy.org/doc/scipy', None),
    # 'matplotlib': ('https://matplotlib.org/stable', None),
    # 'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    # 'skimage': ('https://scikit-image.org/docs/stable', None),
    # 'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    # 'scipy-lecture-notes': ('https://scipy-lectures.org', None),
    # 'pytest': ('https://docs.pytest.org/en/stable', None),
    # 'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
    # 'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
    # 'dlpack': ('https://dmlc.github.io/dlpack/latest', None)
}

skippable_extensions = [
    ('breathe', 'skip generating C/C++ API from comment blocks.'),
]
for ext, warn in skippable_extensions:
    ext_exist = importlib.util.find_spec(ext) is not None
    if ext_exist:
        extensions.append(ext)
    else:
        print(f"Unable to find Sphinx extension '{ext}', {warn}.")


# Make sure the target is unique
autosectionlabel_prefix_document = True

# MathJAX Options
mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# The suffix of source filenames.
source_suffix = '.rst'

# Will change to `root_doc` in Sphinx 4
master_doc = 'index'

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'cloud'
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "image_light": "logo.png",
        # "image_dark": "logo-dark.png",
        # "link": "<other page or external link>",
        # "text": "QuTIpy",
    },
    "github_url": "https://github.com/sumeetkhatri/QuTIpy",
    "twitter_url": "https://twitter.com/sumeetkhatri6",
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", ],

    "external_links": [
        {"name": "Numpy", "url": "https://numpy.org/numpy-tutorials/"},
        {"name": "Book", "url": "https://sumeetkhatri.files.wordpress.com/2021/08/sc_22aug2021.pdf"}
        # {"name": "Principles of Quantum Communication Theory: A Modern Approach", "url": "https://sumeetkhatri.files.wordpress.com/2021/08/sc_22aug2021.pdf"}
    ],

    # "analytics_id": "G-XXXXXXXXXX",  #  Provided by Google in your dashboard
    # "analytics_anonymize_ip": False,

    # Toc options
    "show_toc_level": 1,
    "collapse_navigation": True,
    "show_nav_level": 1,
    "navigation_depth": 4,
}

html_title = f"{project} v{release} Manual"
html_last_updated_fmt = "%b %d, %Y"

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'numpy'

# -----------------------------------------------------------------------------
# Hoverxref extensions
# -----------------------------------------------------------------------------

# hoverxref_auto_ref = True

# hoverxref_roles = [
#     'numref',
#     'confval',
#     'setting',
# ]

hoverxref_role_types = {
    'hoverxref': 'modal',
    'ref': 'modal',  # for hoverxref_auto_ref config
    'confval': 'tooltip',  # for custom object
    'mod': 'tooltip',  # for Python Sphinx Domain
    'class': 'tooltip',  # for Python Sphinx Domain
}

# hoverxref_intersphinx_types = {
#     # make specific links to use a particular tooltip type
#     'readthdocs': {
#         'doc': 'modal',
#         'ref': 'tooltip',
#     },
#     'python': {
#         'class': 'modal',
#         'ref': 'tooltip',
#     },

#     # make all links for Sphinx to be ``tooltip``
#     'sphinx': 'tooltip',
# }

# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = 'dump.xml'

# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True
numpydoc_show_class_members = True
numpydoc_use_blockquotes = True
numpydoc_xref_param_type = True

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True


# -----------------------------------------------------------------------------
# Texinfo output
# -----------------------------------------------------------------------------
_stdauthor = "Written by QuTIpy Community"
texinfo_documents = [
  ("contents", 'qutipy', 'QuTIpy Documentation', _stdauthor, 'QuTIpy',
   "QuTIpy: Quantum Theory of Information for Python.",
   'Programming',
   1),
]

# Options for LaTeX output

# XeLaTeX for better support of unicode characters
latex_engine = 'xelatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # 'fontpkg': r'''
    # \setmainfont{DejaVu Serif}
    # \setsansfont{DejaVu Sans}
    # \setmonofont{DejaVu Sans Mono}
    # ''',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': r'''
    # \usepackage[titles]{tocloft}
    # \cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
    # \setlength{\cftchapnumwidth}{0.75cm}
    # \setlength{\cftsecindent}{\cftchapnumwidth}
    # \setlength{\cftsecnumwidth}{1.25cm}
    # ''',
    # 'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    # 'printindex': r'\footnotesize\raggedright\printindex',
    # 'fontenc': r'\usepackage[LGR,T1]{fontenc}'
}
# latex_show_urls = 'footnote'

# Prevent sphinx-panels from loading bootstrap css, the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False
panels_add_boostrap_css = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = ["numpy.css", "panels_add_bootstrap_css"]
html_static_path = ['_static']



# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'default'
# pygments_style = 'emacs'
# pygments_style = 'friendly'
# pygments_style = 'colorful'
# pygments_style = 'autumn'
# pygments_style = 'murphy'
# pygments_style = 'manni'
# pygments_style = 'material'
# pygments_style = 'monokai,'
# pygments_style = 'perldoc'
# pygments_style = 'pastie'
# pygments_style = 'borland'
# pygments_style = 'trac'
# pygments_style = 'native'
# pygments_style = 'fruity'
# pygments_style = 'bw'
# pygments_style = 'vim'
# pygments_style = 'vs'
# pygments_style = 'tango'
# pygments_style = 'rrt'
# pygments_style = 'xcode'
# pygments_style = 'igor'
# pygments_style = 'paraiso-light'
# pygments_style = 'paraiso-dark'
# pygments_style = 'lovelace'
# pygments_style = 'algol'
# pygments_style = 'algol_nu'
# pygments_style = 'arduino'
# pygments_style = 'rainbow_dash'
# pygments_style = 'abap'
# pygments_style = 'solarized-dark'
# pygments_style = 'solarized-light'
# pygments_style = 'sas'
# pygments_style = 'stata'
# pygments_style = 'stata-light'
# pygments_style = 'stata-dark'
# pygments_style = 'inkpot'
# pygments_style = 'zenburn'
pygments_style = "sphinx"