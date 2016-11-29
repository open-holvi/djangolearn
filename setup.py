#!/usr/bin/python
# -*- coding: utf-8 -*-

import versioneer
from pip.req import parse_requirements

import subprocess
from datetime import datetime

from setuptools import setup, find_packages

# Parses pip requirements and transform this
# into an array to be used at setup time
install_reqs = [str(req.req) for req in
                parse_requirements("./requirements.txt", session={})]

test_reqs = [str(req.req) for req in
                parse_requirements("./test_requirements.txt", session={})]


tests_require = install_reqs + test_reqs

setup(
    name='djangolearn',
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    author='Sergio',
    author_email='sergio@holvi.com',
    packages=find_packages(exclude=('test_djangolearn')),
    install_requires=install_reqs,
    zip_safe=False,
    include_package_data=True,
    test_suite="test_djangolearn.run_tests.run_all",
    tests_require=tests_require,
    setup_requires=['setuptools_scm'],
)
