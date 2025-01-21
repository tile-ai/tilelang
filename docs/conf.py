# -*- coding: utf-8 -*-
import os
import sys

import tlcpack_sphinx_addon

# -- General configuration ------------------------------------------------

sys.path.insert(0, os.path.abspath("../tilelang"))
sys.path.insert(0, os.path.abspath("../"))
autodoc_mock_imports = ["torch"]

# General information about the project.
project = "tilelang"
author = "Tile Lang Contributors"
copyright = "2025-2025, %s" % author

# Version information.

version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_reredirects",
]

redirects = {"get_started/try_out": "../index.html#getting-started"}

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = []

html_static_path = []

footer_copyright = "© 2025-2025 Tile Language"
footer_note = " "

html_logo = "_static/img/logo-row.svg"

html_theme_options = {
    "logo_only": True,
}

header_links = [
    ("Home", "https://github.com/tile-ai/tilelang"),
    ("Github", "https://github.com/tile-ai/tilelang"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "tile-ai",
    "github_repo": "tilelang",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    # "header_logo": "/path/to/logo",
    # "header_logo_link": "",
    # "version_selecter": "",
}

# add additional overrides
templates_path += [tlcpack_sphinx_addon.get_templates_path()]
html_static_path += [tlcpack_sphinx_addon.get_static_path()]
