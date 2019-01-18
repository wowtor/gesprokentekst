#!/usr/bin/env python

from setuptools import setup

setup(name='lrtools',
      version='0.0.1',
      description='LR scripts',
      packages=['liar'],
      setup_requires=['nose'],
      test_suite='nose.collector'
     )
