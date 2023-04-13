#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages

print(find_packages())
setup_args = generate_distutils_setup(     
     packages="pupper_controller",
     package_dir={'': 'src'},     
     py_modules=['src.djipupper', 'src.stanford_lib']
)

print(setup_args)
setup(**setup_args)