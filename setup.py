from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(
    name='tfseqestimator',
    version='1.0.0',
    description='Sequence estimator for Tensorflow',
    url='https://github.com/shkarupa-alex/tfseqestimator',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=['tfseqestimator'],
    install_requires=[
        'tensorflow>=1.8.0',
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
