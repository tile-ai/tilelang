# -*- coding: utf-8 -*-
import os
import sys

# import tlcpack_sphinx_addon

# -- General configuration ------------------------------------------------

sys.path.insert(0, os.path.abspath("../tilelang"))
sys.path.insert(0, os.path.abspath("../"))

autodoc_mock_imports = ["torch", "tilelang.language.ast", "tilelang.language.parser"]

# General information about the project.
project = "Tile Language <br>"
author = "Tile Lang Contributors"
copyright = "2025-2025, %s" % author

# Version information.

# TODO: use the version from project metadata
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_parser",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

redirects = {"get_started/try_out": "../index.html#getting-started"}

source_suffix = [".md", ".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**/*libinfo*", "**/*version*"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"

templates_path = []

html_static_path = ["_static"]

footer_copyright = "© 2025-2025 Tile Language"
footer_note = " "

html_theme_options = {
    "light_logo": "img/logo-row.svg",
    "dark_logo": "img/logo-row.svg",
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

# # add additional overrides
# templates_path += [tlcpack_sphinx_addon.get_templates_path()]
# html_static_path += [tlcpack_sphinx_addon.get_static_path()]
