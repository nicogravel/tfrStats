# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tfrStats'
copyright = 'in dog we trust'
author = 'Nicolas Gravel'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
import alabaster


extensions = []
templates_path = ['_templates']
exclude_patterns = []

extensions = [
    'nbsphinx',
    'myst_parser',
    'alabaster',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.matlab',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_disqus.disqus'
]
typehints_fully_qualified = False
disqus_shortname = 'tfrStats'


myst_enable_extensions = [
    "attrs_image",
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

bibtex_bibfiles = ['references.bib']
bibtex_encoding = 'utf-8-sig'
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'super'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme_path = [alabaster.get_path()]
html_theme = 'alabaster' #'classic' # 'alabaster' #'bizstyle' #
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}


html_static_path = ['_static']
#html_baseurl = '/docs/'
#html_css_files = ['custom.css']
#html_style = 'css/custom.css'

html_theme_options = {
    'logo': 'logo_2.jpg',
    'logo_text_align': 'left',
    'github_banner': True,
    'description': '',
    'description_font_style': 'Caslon',
    'page_width': '100%',
    'body_max_width': 'auto',
    'nosidebar': False,
    'fixed_sidebar': True,
    'sidebar_width': '15%',
    'show_relbars': True,
    'show_powered_by' : 'False',

}

'''


conda activate lamidec



cd /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/
sphinx-apidoc -f -o source /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/tfrStats

rsync -a --delete /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/tfrStats /Users/nicogravel/Documents/Websites/tfrStats/tfrStats/


cd  /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/docs
make clean; make html
rsync -a --delete /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/docs/build/html /Users/nicogravel/Documents/Websites/tfrStats/tfrStats/docs

cd /Users/nicogravel/Documents/Websites/tfrStats/tfrStats/

git add .
git commit -m "shortened the summary text"
git push -u origin main

cd  /Users/nicogravel/Documents/Websites/tfrStats/tfrStats_local/docs


'''
