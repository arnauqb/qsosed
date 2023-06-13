# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='qsosed',
    version='0.2.0',
    description='A physical model for the broadband continuum of Quasars',
    long_description=readme,
    author='Arnau Quera-Bofarull',
    author_email='arnauq@protonmail.com',
    url='https://github.com/arnauqb/qsosed',
    license=license,
    packages=find_packages(exclude=('test*', 'docs')),
    setup_requires=['pbr'],
    pbr=True,
)
