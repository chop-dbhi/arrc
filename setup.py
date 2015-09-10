
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Audiological radiology report classifier modeling and rest service.',
    'author': 'Aaron J. Masino',
    'url': 'URL to get it at.',
    'download_url': 'https://github.com/chop-dbhi/arrc',
    'author_email': 'masinoa@email.chop.edu',
    'version': '0.1',
    'install_requires': ['numpy', 'scikit-learn', 'pandas', 'nltk','flask'],
    'packages': ['learn'],
    'scripts': [],
    'name': 'aarc'
}

setup(**config)
