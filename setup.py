# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='qsosed',
    version='0.1.0',
    description='A physical model for the broadband continuum of Quasars',
    long_description=readme,
    author='Arnau Quera-Bofarull',
    author_email='arnau.quera-bofarull@durham.ac.uk',
    url='https://github.com/arnauq/qsosed',
    license=license,
    packages=find_packages(exclude=('test*', 'docs')),
    setup_requires=['pbr'],
    pbr=True,
)
