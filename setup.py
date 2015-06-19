
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Audiological radiology report classifier modeling and rest service.',
    'author': 'Aaron J. Masino',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'masinoa@email.chop.edu',
    'version': '0.1',
    'install_requires': ['numpy', 'sklearn', 'pandas', 'nltk','flask'],
    'packages': ['learn'],
    'scripts': [],
    'name': 'aarc_service'
}

setup(**config)
