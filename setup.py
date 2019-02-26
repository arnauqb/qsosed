# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyagn',
    version='0.2.0',
    description='PyAGN: A Python code about AGNs.',
    long_description=readme,
    author='Arnau Quera-Bofarull',
    author_email='arnau.quera-bofarull@durham.ac.uk',
    url='https://github.com/arnauq/pyagn',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires = ['numpy',
                        'astropy',
                        'scipy',
                        'matplotlib',
                        'memoized_property']

)

