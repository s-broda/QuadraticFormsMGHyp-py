"""Sphinx configuration for QuadraticFormsMGHyp."""

project = "QuadraticFormsMGHyp"
author = "Simon Broda"
copyright = "2026, Simon Broda"
release = "0.3.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autoclass_content = "class"
autodoc_member_order = "bysource"

html_theme = "furo"
html_title = "QuadraticFormsMGHyp"
html_static_path = []
exclude_patterns = ["_build"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = []
